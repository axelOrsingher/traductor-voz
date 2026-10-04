# Fija el dispositivo de reproducción por defecto (consola, multimedia y comunicaciones)
# usando la interfaz COM no documentada IPolicyConfig. Sin dependencias.
# Uso: .\set_default_audio.ps1 -Match "Speakers"      (substring del nombre)
#      .\set_default_audio.ps1 -List
param([string]$Match = "Realtek", [switch]$List, [switch]$Capture)
$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

[Guid("f8679f50-850a-41cf-9c72-430f290290c8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPolicyConfig {
    int a(); int b(); int c(); int d(); int e(); int f(); int g(); int h(); int i(); int j();
    [PreserveSig] int SetDefaultEndpoint([MarshalAs(UnmanagedType.LPWStr)] string id, int role);
}
[ComImport, Guid("870af99c-171d-4f9e-af0d-e63df40c2bc9")] class PolicyConfigClient {}

[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(int flow, int mask, out IMMDeviceCollection col);
    int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice dev);
}
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumerator {}
[Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceCollection { int GetCount(out int n); int Item(int i, out IMMDevice dev); }
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
    int Activate(); int OpenPropertyStore(int access, out IPropertyStore ps);
    int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
}
[Guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPropertyStore { int GetCount(out int n); int GetAt(int i, out PROPERTYKEY k); int GetValue(ref PROPERTYKEY k, out PROPVARIANT v); }
[StructLayout(LayoutKind.Sequential)] public struct PROPERTYKEY { public Guid fmtid; public int pid; }
[StructLayout(LayoutKind.Explicit)] public struct PROPVARIANT { [FieldOffset(0)] public short vt; [FieldOffset(8)] public IntPtr p; [FieldOffset(16)] public IntPtr p2; }

public static class AudioDefault {
    public static List<string[]> List(int flow) {
        var r = new List<string[]>();
        var en = (IMMDeviceEnumerator)(new MMDeviceEnumerator());
        IMMDeviceCollection col; en.EnumAudioEndpoints(flow, 1, out col);
        int n; col.GetCount(out n);
        string defId = null;
        try { IMMDevice d; en.GetDefaultAudioEndpoint(flow, 0, out d); d.GetId(out defId); } catch {}
        for (int i = 0; i < n; i++) {
            IMMDevice dev; col.Item(i, out dev);
            string id; dev.GetId(out id);
            IPropertyStore ps; dev.OpenPropertyStore(0, out ps);
            var key = new PROPERTYKEY { fmtid = new Guid("a45c254e-df1c-4efd-8020-67d146a850e0"), pid = 14 };
            PROPVARIANT v; ps.GetValue(ref key, out v);
            r.Add(new[] { id, Marshal.PtrToStringUni(v.p), id == defId ? "*" : " " });
        }
        return r;
    }
    public static void Set(string id) {
        var pc = (IPolicyConfig)(new PolicyConfigClient());
        for (int role = 0; role < 3; role++) Marshal.ThrowExceptionForHR(pc.SetDefaultEndpoint(id, role));
    }
}
"@

$flow = if ($Capture) { 1 } else { 0 }
$devs = [AudioDefault]::List($flow)
if ($List) { $devs | ForEach-Object { "{0} {1}" -f $_[2], $_[1] }; return }
$hit = @($devs | Where-Object { $_[1] -like "*$Match*" })
if ($hit.Count -ne 1) { throw "'$Match' coincide con $($hit.Count) dispositivos: $(($devs | ForEach-Object { $_[1] }) -join ' | ')" }
[AudioDefault]::Set($hit[0][0])
"Default -> $($hit[0][1])"
