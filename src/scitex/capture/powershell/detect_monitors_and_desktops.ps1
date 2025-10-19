# Detect Monitors and Virtual Desktops in Windows
# Returns comprehensive information about physical monitors and virtual desktop state

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Get all physical monitors
$screens = [System.Windows.Forms.Screen]::AllScreens
$monitorInfo = @()

foreach ($screen in $screens) {
    $bounds = $screen.Bounds
    $monitorInfo += @{
        DeviceName = $screen.DeviceName
        IsPrimary = $screen.Primary
        Bounds = @{
            X = $bounds.X
            Y = $bounds.Y
            Width = $bounds.Width
            Height = $bounds.Height
        }
        WorkingArea = @{
            X = $screen.WorkingArea.X
            Y = $screen.WorkingArea.Y
            Width = $screen.WorkingArea.Width
            Height = $screen.WorkingArea.Height
        }
        BitsPerPixel = $screen.BitsPerPixel
    }
}

# Virtual Desktop detection (Windows 10/11)
# Note: Virtual Desktop API is not officially exposed via PowerShell
# This uses a workaround to detect virtual desktops through Task View
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class VirtualDesktopDetector {
    [DllImport("user32.dll")]
    public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc enumProc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll", SetLastError = true)]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
}
"@

# Count visible windows (proxy for virtual desktop activity)
$visibleWindows = 0
$windowList = @()

$callback = {
    param($hWnd, $lParam)

    if ([VirtualDesktopDetector]::IsWindowVisible($hWnd)) {
        $length = [VirtualDesktopDetector]::GetWindowTextLength($hWnd)
        if ($length -gt 0) {
            $sb = New-Object System.Text.StringBuilder($length + 1)
            [VirtualDesktopDetector]::GetWindowText($hWnd, $sb, $sb.Capacity) | Out-Null
            $title = $sb.ToString()

            if ($title -and $title -ne "") {
                $processId = 0
                [VirtualDesktopDetector]::GetWindowThreadProcessId($hWnd, [ref]$processId) | Out-Null

                $processName = "Unknown"
                try {
                    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                    if ($process) {
                        $processName = $process.ProcessName
                    }
                } catch {
                    # Process may have exited, use Unknown
                }

                # Always add window, even if process name unknown
                $script:windowList += @{
                    Handle = $hWnd.ToInt64()
                    Title = $title
                    ProcessId = $processId
                    ProcessName = $processName
                }
                $script:visibleWindows++
            }
        }
    }
    return $true
}

[VirtualDesktopDetector]::EnumWindows($callback, [IntPtr]::Zero)

# Check if Task View / Virtual Desktop feature is enabled
$taskViewEnabled = $false
$virtualDesktopCount = 1  # Default to 1 (always have at least one desktop)

try {
    # Try to detect virtual desktop support through registry
    $regPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced"
    $taskViewKey = Get-ItemProperty -Path $regPath -Name "TaskView" -ErrorAction SilentlyContinue
    if ($taskViewKey) {
        $taskViewEnabled = $true
    }
} catch {
    # Virtual desktop support detection failed
}

# Build output
$result = @{
    Monitors = @{
        Count = $monitorInfo.Count
        Details = $monitorInfo
        PrimaryMonitor = ($monitorInfo | Where-Object { $_.IsPrimary -eq $true }).DeviceName
    }
    VirtualDesktops = @{
        Supported = $taskViewEnabled
        EstimatedCount = $virtualDesktopCount
        Note = "Windows does not officially expose Virtual Desktop count via PowerShell. Count represents minimum (current desktop)."
    }
    Windows = @{
        VisibleCount = $visibleWindows
        TotalEnumerated = $windowList.Count
        Details = $windowList
    }
    Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}

# Output as JSON
$result | ConvertTo-Json -Depth 5 -Compress
