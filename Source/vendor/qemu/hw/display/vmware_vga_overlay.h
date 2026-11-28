#ifndef HW_DISPLAY_VMWARE_VGA_OVERLAY_H
#define HW_DISPLAY_VMWARE_VGA_OVERLAY_H

/* Show the DXVK SDL window as a click-through child overlay of QEMU's SDL
 * console window, sized to its content area.  Both are SDL_Window *. */
void vmsvga_overlay_show(void *dxvk_window, void *qemu_window);
void vmsvga_overlay_hide(void *dxvk_window);
/* Last fitted overlay size in physical pixels (0,0 until first fit). */
void vmsvga_overlay_pixel_size(int *w, int *h);

/* Guest resolution of the zero-copy swapchain (drawable size to keep). */
void vmsvga_overlay_set_source_size(int w, int h);

/* Standalone headless CAMetalLayer for VK_EXT_metal_surface (no AppKit/SDL
 * window required; safe under -display cocoa, -display sdl, and -display none). */
void *vmsvga_metal_layer_create(int w, int h);
void vmsvga_metal_layer_resize(void *layer, int w, int h);
void vmsvga_metal_layer_destroy(void *layer);

#endif
