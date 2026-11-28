#ifndef HW_I386_VMPORT_VMVGA_H
#define HW_I386_VMPORT_VMVGA_H

typedef bool VMPortSVGACapabilityFunc(void *opaque, uint32_t type,
                                      uint32_t *value);

bool vmport_register_svga_capability_provider(
    VMPortSVGACapabilityFunc *func, void *opaque);

/* Queue a host -> guest VMware Tools TCLO command, e.g.
 * "Resolution_Set 1280 800".  Resolution requests are coalesced. */
void vmport_tclo_queue(const char *command);

void vmport_guestinfo_set(const char *key, const char *value);

#endif
