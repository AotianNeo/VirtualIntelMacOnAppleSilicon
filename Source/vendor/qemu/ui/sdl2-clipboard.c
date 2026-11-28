/*
 * SDL2 clipboard integration.
 *
 * QEMU's clipboard core (qemu_clipboard_*) is UI-agnostic, but only the Cocoa
 * and VNC front ends ever registered a peer - so host<->guest clipboard did not
 * work under -display sdl. That matters here because the qemu-3dfx Glide/Mesa
 * pass-through REQUIRES -display sdl (it blits straight to the SDL2 window), so
 * users had to choose between 3D acceleration and clipboard sharing.
 *
 * SDL2 has had SDL_GetClipboardText/SDL_SetClipboardText and the
 * SDL_CLIPBOARDUPDATE event since 2.0.0, so wiring a peer up is straightforward.
 */
#include "qemu/osdep.h"
#include "qemu/main-loop.h"
#include "qemu/timer.h"
#include "ui/console.h"
#include "ui/clipboard.h"
#include "ui/sdl2.h"

static QemuClipboardInfo *sdl2_cbinfo;      /* what the host currently offers */
static QemuEvent          sdl2_cbevent;
static bool               sdl2_cb_ready;
static char              *sdl2_last_text;   /* echo suppression + macOS poll */
static int64_t            sdl2_last_poll_ms;

static void sdl2_clipboard_notify(Notifier *notifier, void *data);
static void sdl2_clipboard_request(QemuClipboardInfo *info,
                                   QemuClipboardType type);

static QemuClipboardPeer sdl2_cbpeer = {
    .name     = "sdl2",
    .notifier = { .notify = sdl2_clipboard_notify },
    .request  = sdl2_clipboard_request,
};

/*
 * Guest -> host. The guest agent published new clipboard contents; push the
 * text into SDL so the host desktop can paste it.
 */
static void sdl2_clipboard_notify(Notifier *notifier, void *data)
{
    QemuClipboardNotify *notify = data;

    switch (notify->type) {
    case QEMU_CLIPBOARD_UPDATE_INFO: {
        QemuClipboardInfo *info = notify->info;

        if (info->selection != QEMU_CLIPBOARD_SELECTION_CLIPBOARD) {
            return;
        }
        if (info->owner == &sdl2_cbpeer) {
            /* We published this ourselves - do not echo it back. */
            return;
        }
        if (info != sdl2_cbinfo) {
            qemu_clipboard_info_unref(sdl2_cbinfo);
            sdl2_cbinfo = qemu_clipboard_info_ref(info);
        }
        if (info->types[QEMU_CLIPBOARD_TYPE_TEXT].available &&
            info->types[QEMU_CLIPBOARD_TYPE_TEXT].data == NULL) {
            /* Text offered but not yet fetched: ask the guest for it. */
            qemu_clipboard_request(info, QEMU_CLIPBOARD_TYPE_TEXT);
            return;
        }
        if (info->types[QEMU_CLIPBOARD_TYPE_TEXT].data) {
            g_autofree char *text =
                g_strndup(info->types[QEMU_CLIPBOARD_TYPE_TEXT].data,
                          info->types[QEMU_CLIPBOARD_TYPE_TEXT].size);
            /* Record before SDL_SetClipboardText so the synthetic
             * SDL_CLIPBOARDUPDATE emitted by sdl2-compat/SDL3 on macOS does
             * not bounce ownership back to the host. */
            g_free(sdl2_last_text);
            sdl2_last_text = g_strdup(text);
            SDL_SetClipboardText(text);
        }
        break;
    }
    case QEMU_CLIPBOARD_RESET_SERIAL:
        break;
    default:
        break;
    }
}

/*
 * Host -> guest. The guest asked for the data behind an offer we made; hand it
 * whatever SDL currently holds.
 */
static void sdl2_clipboard_request(QemuClipboardInfo *info,
                                   QemuClipboardType type)
{
    char *text;

    if (type != QEMU_CLIPBOARD_TYPE_TEXT) {
        return;
    }
    text = SDL_GetClipboardText();
    if (text == NULL) {
        return;
    }
    if (*text) {
        qemu_clipboard_set_data(&sdl2_cbpeer, info, type,
                                strlen(text), text, true);
    }
    SDL_free(text);
}

/*
 * Called from the SDL event loop on SDL_CLIPBOARDUPDATE or periodic poll:
 * if the host clipboard text actually changed, advertise text availability
 * to the guest. The data itself is fetched lazily via sdl2_clipboard_request().
 */
void sdl2_clipboard_handle_update(void)
{
    QemuClipboardInfo *info;
    char *text;

    if (!sdl2_cb_ready || !SDL_HasClipboardText()) {
        return;
    }
    text = SDL_GetClipboardText();
    if (text == NULL) {
        return;
    }
    if (!*text || g_strcmp0(text, sdl2_last_text) == 0) {
        SDL_free(text);
        return;
    }
    g_free(sdl2_last_text);
    sdl2_last_text = g_strdup(text);
    SDL_free(text);

    info = qemu_clipboard_info_new(&sdl2_cbpeer,
                                   QEMU_CLIPBOARD_SELECTION_CLIPBOARD);
    info->types[QEMU_CLIPBOARD_TYPE_TEXT].available = true;
    qemu_clipboard_info_unref(sdl2_cbinfo);
    sdl2_cbinfo = info;
    qemu_clipboard_update(info);
}

/*
 * macOS NSPasteboard has no push notification for external clipboard changes
 * (e.g. pbcopy or Cmd+C in another app), and sdl2-compat does not emit
 * SDL_CLIPBOARDUPDATE for them. Poll periodically and on window focus gain.
 */
void sdl2_clipboard_poll(bool force)
{
    int64_t now;

    if (!sdl2_cb_ready) {
        return;
    }
    now = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
    if (!force && now - sdl2_last_poll_ms < 250) {
        return;
    }
    sdl2_last_poll_ms = now;
    sdl2_clipboard_handle_update();
}

void sdl2_clipboard_init(void)
{
    if (sdl2_cb_ready) {
        return;
    }
    qemu_event_init(&sdl2_cbevent, false);
    qemu_clipboard_peer_register(&sdl2_cbpeer);
    sdl2_cb_ready = true;
}
