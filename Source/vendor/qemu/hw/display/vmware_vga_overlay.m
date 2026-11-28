#include <stdio.h>
/*
 * vmvga zero-copy overlay (macOS).
 *
 * DXVK renders into its own (normally hidden) SDL window.  For zero-copy
 * presentation that window is shown borderless on top of QEMU's SDL console
 * window, as a click-through NSWindow child, so it follows the parent's
 * position and size while keyboard/mouse input keeps going to QEMU.
 *
 * All AppKit work is dispatched to the main thread: FIFO processing (and so
 * the caller) may run on a vCPU thread.
 */
#include <SDL.h>
#include <SDL_syswm.h>
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include "vmware_vga_overlay.h"

void *vmsvga_metal_layer_create(int w, int h)
{
    @autoreleasepool {
        CAMetalLayer *layer = [[CAMetalLayer alloc] init];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev != nil) {
            [layer setDevice:dev];
            [dev release];
        }
        [layer setPixelFormat:MTLPixelFormatBGRA8Unorm];
        [layer setContentsScale:1.0];
        [layer setFrame:CGRectMake(0, 0, w > 0 ? w : 640, h > 0 ? h : 480)];
        [layer setDrawableSize:CGSizeMake(w > 0 ? w : 640, h > 0 ? h : 480)];
        return (void *)layer;
    }
}

void vmsvga_metal_layer_resize(void *layer_ptr, int w, int h)
{
    if (layer_ptr == NULL || w <= 0 || h <= 0) {
        return;
    }
    @autoreleasepool {
        CAMetalLayer *layer = (CAMetalLayer *)layer_ptr;
        [layer setFrame:CGRectMake(0, 0, w, h)];
        [layer setDrawableSize:CGSizeMake(w, h)];
    }
}

void vmsvga_metal_layer_destroy(void *layer_ptr)
{
    if (layer_ptr == NULL) {
        return;
    }
    @autoreleasepool {
        CAMetalLayer *layer = (CAMetalLayer *)layer_ptr;
        [layer release];
    }
}

/* Real pixel size of the overlay content, updated on the main thread each
 * fit and read by the DXVK shim (any thread) when sizing the swapchain. */
static volatile int vmsvga_overlay_px_w, vmsvga_overlay_px_h;
/* Zero-copy swapchain size (guest resolution), set by the device. */
static volatile int vmsvga_overlay_src_w, vmsvga_overlay_src_h;

void vmsvga_overlay_set_source_size(int w, int h)
{
    vmsvga_overlay_src_w = w;
    vmsvga_overlay_src_h = h;
}

void vmsvga_overlay_pixel_size(int *w, int *h)
{
    *w = vmsvga_overlay_px_w;
    *h = vmsvga_overlay_px_h;
}

static NSWindow *vmsvga_overlay_nswindow(SDL_Window *window)
{
    SDL_SysWMinfo info;

    if (window == NULL) {
        return nil;
    }
    SDL_VERSION(&info.version);
    if (!SDL_GetWindowWMInfo(window, &info) ||
        info.subsystem != SDL_SYSWM_COCOA) {
        return nil;
    }
    return info.info.cocoa.window;
}

static CAMetalLayer *vmsvga_overlay_metal_layer(NSView *view)
{
    if (view == nil) {
        return nil;
    }
    if ([[view layer] isKindOfClass:[CAMetalLayer class]]) {
        return (CAMetalLayer *)[view layer];
    }
    for (NSView *sub in [view subviews]) {
        CAMetalLayer *layer = vmsvga_overlay_metal_layer(sub);
        if (layer != nil) {
            return layer;
        }
    }
    return nil;
}

/* Keep the overlay exactly on the parent's content area, and keep the
 * Metal drawable in physical (Retina) pixels of that area.  MoltenVK reads
 * the swapchain extent from drawableSize, so DXVK recreates its swapchain
 * at the new size; contentsGravity scales any frame still in flight. */
SDL_Renderer *sdl2_vmvga_primary_renderer(void);

/* Place the overlay exactly where QEMU draws the guest image: the
 * logical-size viewport of QEMU's own SDL renderer (ui/sdl2-2d.c), which is
 * also what SDL uses to map mouse events.  No geometry of our own. */
static void vmsvga_overlay_fit(NSWindow *child, NSWindow *parent,
                               SDL_Window *parent_sdl)
{
    static NSRect last;
    NSRect content = [parent contentRectForFrameRect:[parent frame]];
    SDL_Renderer *r = sdl2_vmvga_primary_renderer();
    CAMetalLayer *layer = vmsvga_overlay_metal_layer([child contentView]);
    SDL_Rect vp;
    float sx = 1.0f, sy = 1.0f, pts;
    int ow = 0, oh = 0, ww = 0, wh = 0;
    CGFloat rx, ry, rw, rh;
    NSRect target;
    CGSize d;

    vmsvga_overlay_px_w = (int)(content.size.width * [parent backingScaleFactor]);
    vmsvga_overlay_px_h = (int)(content.size.height * [parent backingScaleFactor]);
    if (r == NULL || layer == nil ||
        SDL_GetRendererOutputSize(r, &ow, &oh) != 0 || ow <= 0 || oh <= 0) {
        return;
    }
    SDL_GetWindowSize(parent_sdl, &ww, &wh);
    SDL_RenderGetViewport(r, &vp);
    SDL_RenderGetScale(r, &sx, &sy);
    if (ww <= 0 || vp.w <= 0 || vp.h <= 0) {
        return;
    }
    pts = (float)ww / (float)ow;              /* renderer pixels -> points */
    rx = vp.x * sx * pts;
    ry = vp.y * sy * pts;
    rw = vp.w * sx * pts;
    rh = vp.h * sy * pts;
    /* SDL: top-left origin; Cocoa: bottom-left origin */
    target = NSMakeRect(content.origin.x + rx,
                        content.origin.y + content.size.height - (ry + rh),
                        rw, rh);
    if (!NSEqualRects([child frame], target)) {
        [child setFrame:target display:NO];
    }
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    for (NSView *sub in [[child contentView] subviews]) {
        [sub setFrame:[[child contentView] bounds]];
    }
    [layer setFrame:[[child contentView] bounds]];
    /* SDL's Metal view resizes the drawable to its bounds on every layout;
     * DXVK renders at the guest resolution, so force the drawable back to
     * exactly that (after the layout above) and scale it to the overlay. */
    if (vmsvga_overlay_src_w > 0 && vmsvga_overlay_src_h > 0) {
        CGSize want = CGSizeMake(vmsvga_overlay_src_w, vmsvga_overlay_src_h);
        if (!CGSizeEqualToSize([layer drawableSize], want)) {
            [layer setDrawableSize:want];
        }
    }
    d = [layer drawableSize];
    if (d.width >= 1 && rw >= 1) {
        [layer setContentsScale:d.width / rw];
    }
    [layer setContentsGravity:kCAGravityResize];
    [CATransaction commit];
    if (!NSEqualRects(last, target)) {
        last = target;
        fprintf(stderr, "VMVGA-ZERO-COPY geometry viewport=%d,%d %dx%d "
                "scale=%.3f,%.3f out=%dx%d win=%dx%d pts=%.3f "
                "overlay=%.0f,%.0f %.0fx%.0f drawable=%.0fx%.0f\n",
                vp.x, vp.y, vp.w, vp.h, sx, sy, ow, oh, ww, wh, pts,
                rx, ry, rw, rh, d.width, d.height);
    }
}

void vmsvga_overlay_show(void *dxvk_window, void *qemu_window)
{
    SDL_Window *child_sdl = dxvk_window;
    SDL_Window *parent_sdl = qemu_window;

    dispatch_async(dispatch_get_main_queue(), ^{
        NSWindow *child = vmsvga_overlay_nswindow(child_sdl);
        NSWindow *parent = vmsvga_overlay_nswindow(parent_sdl);

        if (child == nil || parent == nil) {
            return;
        }
        if ([child parentWindow] != parent) {
            [child setStyleMask:NSWindowStyleMaskBorderless];
            [child setIgnoresMouseEvents:YES];
            [child setHasShadow:NO];
            [child setOpaque:YES];
            [child setCollectionBehavior:
                NSWindowCollectionBehaviorFullScreenAuxiliary];
            [parent addChildWindow:child ordered:NSWindowAbove];
        }
        vmsvga_overlay_fit(child, parent, parent_sdl);
        if (![child isVisible]) {
            [child orderFront:nil];
        }
    });
}

void vmsvga_overlay_hide(void *dxvk_window)
{
    SDL_Window *child_sdl = dxvk_window;

    dispatch_async(dispatch_get_main_queue(), ^{
        NSWindow *child = vmsvga_overlay_nswindow(child_sdl);

        if (child != nil && [child isVisible]) {
            [child orderOut:nil];
        }
    });
}
