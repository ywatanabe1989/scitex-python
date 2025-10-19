# Capture a specific window by its handle
# Allows selective capture of individual application windows

param(
    [Parameter(Mandatory=$true)]
    [long]$WindowHandle,

    [Parameter(Mandatory=$false)]
    [string]$OutputFormat = "base64"  # "base64" or "file"
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Enable DPI awareness
Add-Type @'
using System;
using System.Runtime.InteropServices;

public class WindowCapture {
    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint nFlags);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder lpString, int nMaxCount);

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
}
'@

$null = [WindowCapture]::SetProcessDPIAware()

try {
    $hWnd = [IntPtr]$WindowHandle

    # Check if window is visible
    if (-not [WindowCapture]::IsWindowVisible($hWnd)) {
        Write-Error "Window is not visible or does not exist"
        exit 1
    }

    # Get window rectangle
    $rect = New-Object WindowCapture+RECT
    if (-not [WindowCapture]::GetWindowRect($hWnd, [ref]$rect)) {
        Write-Error "Failed to get window rectangle"
        exit 1
    }

    # Calculate dimensions
    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top

    if ($width -le 0 -or $height -le 0) {
        Write-Error "Invalid window dimensions: ${width}x${height}"
        exit 1
    }

    # Get window title
    $sb = New-Object System.Text.StringBuilder(256)
    [WindowCapture]::GetWindowText($hWnd, $sb, $sb.Capacity) | Out-Null
    $windowTitle = $sb.ToString()

    # Create bitmap
    $bitmap = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $hdc = $graphics.GetHdc()

    # Capture window
    $success = [WindowCapture]::PrintWindow($hWnd, $hdc, 2)  # 2 = PW_RENDERFULLCONTENT

    $graphics.ReleaseHdc($hdc)

    if (-not $success) {
        Write-Error "PrintWindow failed"
        $graphics.Dispose()
        $bitmap.Dispose()
        exit 1
    }

    # Convert to base64
    $stream = New-Object System.IO.MemoryStream
    $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
    $bytes = $stream.ToArray()
    $base64 = [Convert]::ToBase64String($bytes)

    # Output result
    $result = @{
        WindowHandle = $WindowHandle
        WindowTitle = $windowTitle
        Width = $width
        Height = $height
        Success = $true
        Base64Data = $base64
        Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }

    $result | ConvertTo-Json -Depth 3 -Compress

    # Cleanup
    $graphics.Dispose()
    $bitmap.Dispose()
    $stream.Dispose()

} catch {
    $errorResult = @{
        WindowHandle = $WindowHandle
        Success = $false
        Error = $_.Exception.Message
        Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }

    $errorResult | ConvertTo-Json -Depth 2 -Compress
    exit 1
}
