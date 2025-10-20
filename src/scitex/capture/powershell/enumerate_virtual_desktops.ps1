# Enumerate Windows Virtual Desktops
# This script detects and lists all virtual desktops in Windows 10/11

Add-Type @"
using System;
using System.Runtime.InteropServices;

public class VirtualDesktop {
    [DllImport("user32.dll")]
    public static extern IntPtr GetDesktopWindow();

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc enumProc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern int GetWindowTextLength(IntPtr hWnd);

    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
}
"@

# Get all visible windows
$windows = @()
$callback = {
    param($hWnd, $lParam)

    if ([VirtualDesktop]::IsWindowVisible($hWnd)) {
        $length = [VirtualDesktop]::GetWindowTextLength($hWnd)
        if ($length -gt 0) {
            $sb = New-Object System.Text.StringBuilder($length + 1)
            [VirtualDesktop]::GetWindowText($hWnd, $sb, $sb.Capacity) | Out-Null
            $title = $sb.ToString()

            if ($title) {
                $windows += @{
                    Handle = $hWnd.ToInt64()
                    Title = $title
                }
            }
        }
    }
    return $true
}

[VirtualDesktop]::EnumWindows($callback, [IntPtr]::Zero)

# Output as JSON for easy parsing
$result = @{
    TotalWindows = $windows.Count
    Windows = $windows
    Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}

$result | ConvertTo-Json -Depth 3
