#!/usr/bin/env python3
"""Pause and resume QEMU while HVDBT is executing a linked FEX loop."""

from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time

tests = Path(__file__).resolve().parent
spec = spec_from_file_location("ext_diff", tests / "test-ext-diff.py")
fixture = module_from_spec(spec)
spec.loader.exec_module(fixture)

loop = [("linked_fex_pause", """
    mov al, 'L'
    out 0xe9, al
    jmp fex_pause_left

    align 512
fex_pause_left:
    inc r8d
    jmp fex_pause_right

    align 512
fex_pause_right:
    inc r9d
    jmp fex_pause_left
""")]


def qmp(stream, command):
    stream.write((json.dumps({"execute": command}) + "\n").encode())
    for line in stream:
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        if "return" in response:
            return response["return"]
    raise RuntimeError("QMP closed before replying")


def test(binary, tmp):
    asm = tmp / "guest.asm"
    obj = tmp / "guest.o"
    elf = tmp / "guest.elf"
    layout = tmp / "guest.ld"
    asm.write_text(fixture.build_asm(loop))
    layout.write_text("ENTRY(_start)\nSECTIONS { . = 0x100000; .text : { *(.text*) } .bss : { *(.bss*) } }\n")
    subprocess.run([fixture.NASM, "-f", "elf32", asm, "-o", obj], check=True)
    subprocess.run([fixture.LLD, "-m", "elf_i386", "-N", "-T", layout, obj, "-o", elf], check=True)
    symbols = subprocess.check_output(["nm", elf], text=True)
    loop_rips = {
        int(address, 16) | 0xffff800000000000
        for address in re.findall(r"^([0-9a-fA-F]+)\s+\w\s+fex_pause_(?:left|right)$", symbols, re.M)
    }
    assert len(loop_rips) == 2, symbols

    serial = tmp / "guest.log"
    monitor = tmp / "qmp.sock"
    log_path = tmp / "qemu.log"
    command = [
        binary, "-M", "q35,accel=tcg", "-cpu", "Skylake-Client-v3,+avx,+avx2,+x2apic",
        "-smp", "2", "-m", "256M", "-display", "none", "-no-reboot",
        "-chardev", f"file,path={serial},id=dbg", "-device", "isa-debugcon,iobase=0xe9,chardev=dbg",
        "-qmp", f"unix:{monitor},server,nowait", "-kernel", elf,
        "-L", tests.parents[1] / "vendor/qemu/pc-bios",
    ]
    env = os.environ | {
        "LC_HVDBT": "1", "LC_HVDBT_FAST_FLAGS": "1", "LC_HVDBT_FEX": "2",
        "LC_HVDBT_FEX_NO_MULTIBLOCK": "1", "LC_HVDBT_FEX_NO_CHAIN": "0", "LC_HVDBT_FEX_TRACE": "1",
    }
    with log_path.open("wb") as log:
        vm = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            while vm.poll() is None and time.monotonic() < deadline:
                output = log_path.read_text(errors="replace")
                compiled = {int(rip, 16) for rip in re.findall(r"compiled rip=0x([0-9a-f]+)", output)}
                if serial.exists() and b"L" in serial.read_bytes() and loop_rips <= compiled:
                    break
                time.sleep(0.02)
            else:
                raise RuntimeError("The guest did not compile both loop blocks with FEX")

            with socket.socket(socket.AF_UNIX) as sock:
                sock.settimeout(3)
                sock.connect(str(monitor))
                with sock.makefile("rwb", buffering=0) as stream:
                    stream.readline()
                    qmp(stream, "qmp_capabilities")
                    for attempt in range(1, 4):
                        started = time.monotonic()
                        qmp(stream, "stop")
                        assert not qmp(stream, "query-status")["running"], "The vCPUs did not pause"
                        print(f"Pause {attempt} completed in {time.monotonic() - started:.3f}s")
                        qmp(stream, "cont")
                        assert qmp(stream, "query-status")["running"], "The vCPUs did not resume"
                        time.sleep(0.1)
                    qmp(stream, "quit")
                    vm.wait(timeout=3)
        except Exception:
            print(log_path.read_text(errors="replace")[-3000:], file=sys.stderr)
            raise
        finally:
            if vm.poll() is None:
                vm.kill()
                vm.wait()


with tempfile.TemporaryDirectory(prefix="fex-pause-", dir="/tmp") as temporary:
    test(sys.argv[1] if len(sys.argv) > 1 else fixture.QEMU, Path(temporary))
