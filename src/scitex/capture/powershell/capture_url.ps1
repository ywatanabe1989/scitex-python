# Capture URL screenshot by opening browser, capturing window, then closing
# Works with Windows host URLs from WSL with proper rendering

param(
    [Parameter(Mandatory=$true)]
    [string]$Url,

    [Parameter(Mandatory=$false)]
    [int]$WaitSeconds = 3,

    [Parameter(Mandatory=$false)]
    [string]$OutputFormat = "base64",

    [Parameter(Mandatory=$false)]
    [int]$WindowWidth = 1920,

    [Parameter(Mandatory=$false)]
    [int]$WindowHeight = 1080
)

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class WindowHelper {
    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();

    [DllImport("user32.dll")]
    public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint nFlags);

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll")]
    public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);

    [DllImport("user32.dll")]
    public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
}
"@

$null = [WindowHelper]::SetProcessDPIAware()

try {
    # Open URL in default browser
    $process = Start-Process $Url -PassThru

    # Wait for page to load and render
    Start-Sleep -Seconds $WaitSeconds

    # Find the browser window
    $browserHandle = $null
    $browserProcesses = @('chrome', 'msedge', 'firefox', 'brave')

    foreach ($browserName in $browserProcesses) {
        $processes = Get-Process -Name $browserName -ErrorAction SilentlyContinue
        if ($processes) {
            # Get the most recently active window
            foreach ($proc in $processes) {
                if ($proc.MainWindowHandle -ne [IntPtr]::Zero) {
                    $browserHandle = $proc.MainWindowHandle
                    break
                }
            }
            if ($browserHandle) { break }
        }
    }

    if (-not $browserHandle) {
        throw "Browser window not found"
    }

    # Resize window to specified dimensions for consistent layout
    # SW_RESTORE = 9 (restore if minimized/maximized)
    [WindowHelper]::ShowWindow($browserHandle, 9) | Out-Null
    Start-Sleep -Milliseconds 200

    # Move and resize window (X=0, Y=0, Width, Height, Repaint=true)
    [WindowHelper]::MoveWindow($browserHandle, 0, 0, $WindowWidth, $WindowHeight, $true) | Out-Null
    Start-Sleep -Milliseconds 300

    # Bring browser to foreground for better capture
    [WindowHelper]::SetForegroundWindow($browserHandle) | Out-Null
    Start-Sleep -Milliseconds 500

    # Get window rectangle
    $rect = New-Object WindowHelper+RECT
    [WindowHelper]::GetWindowRect($browserHandle, [ref]$rect) | Out-Null

    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top

    if ($width -le 0 -or $height -le 0) {
        throw "Invalid window size: ${width}x${height}"
    }

    # Capture window with proper rendering
    $bitmap = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)

    # Set high quality
    $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic

    $hdc = $graphics.GetHdc()

    # PW_RENDERFULLCONTENT = 2
    [WindowHelper]::PrintWindow($browserHandle, $hdc, 2) | Out-Null

    $graphics.ReleaseHdc($hdc)

    # Convert to base64
    $stream = New-Object System.IO.MemoryStream
    $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
    $bytes = $stream.ToArray()
    $base64 = [Convert]::ToBase64String($bytes)

    $result = @{
        Success = $true
        Url = $Url
        Width = $bitmap.Width
        Height = $bitmap.Height
        Base64Data = $base64
        Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }

    $result | ConvertTo-Json -Depth 2 -Compress

    # Cleanup
    $webBrowser.Dispose()
    $bitmap.Dispose()
    $stream.Dispose()

} catch {
    $errorResult = @{
        Success = $false
        Url = $Url
        Error = $_.Exception.Message
        Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }

    $errorResult | ConvertTo-Json -Depth 2 -Compress
    exit 1
}

