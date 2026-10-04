"""Paso 3: enumerar dispositivos WASAPI (incluye loopback y VB-Cable)."""

import pyaudiowpatch as pyaudio

with pyaudio.PyAudio() as pa:
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    print(f"WASAPI host api #{wasapi['index']}")
    default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    default_in = pa.get_device_info_by_index(wasapi["defaultInputDevice"])
    print(f"default out: [{default_out['index']}] {default_out['name']}")
    print(f"default in : [{default_in['index']}] {default_in['name']}")
    print()
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d["hostApi"] != wasapi["index"]:
            continue
        kind = "LOOPBACK" if d.get("isLoopbackDevice") else ("IN " if d["maxInputChannels"] else "OUT")
        print(f"[{i:2}] {kind:8} {int(d['defaultSampleRate'])} Hz "
              f"in={d['maxInputChannels']} out={d['maxOutputChannels']}  {d['name']}")
