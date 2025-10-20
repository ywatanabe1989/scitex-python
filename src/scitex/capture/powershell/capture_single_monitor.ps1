param(
    [Parameter(Mandatory=$false)]
    [int]$MonitorNumber = 0,  # 0-based index from Python
    
    [Parameter(Mandatory=$false)]
    [string]$OutputFormat = "base64"  # "base64" or "file"
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Enable DPI awareness for proper high-resolution capture
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class User32 {
    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
    
    [DllImport("shcore.dll")]
    public static extern int GetDpiForMonitor(IntPtr hmonitor, int dpiType, out uint dpiX, out uint dpiY);
    
    [DllImport("user32.dll")]
    public static extern IntPtr MonitorFromWindow(IntPtr hwnd, uint dwFlags);
}
'@

# Set process DPI aware
$null = [User32]::SetProcessDPIAware()

# Get all screens
$screens = [System.Windows.Forms.Screen]::AllScreens

# Monitor number (0-based index from Python)
$monitorIndex = $MonitorNumber

# Check if monitor exists
if ($monitorIndex -ge $screens.Count -or $monitorIndex -lt 0) {
    Write-Error "Monitor $MonitorNumber not found. Valid range: 0-$(($screens.Count - 1)). $($screens.Count) monitor(s) available."
    exit 1
}

# Get the specified monitor
$targetScreen = $screens[$monitorIndex]
$bounds = $targetScreen.Bounds

# Create bitmap with screen dimensions
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)

# Set high quality rendering
$graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
$graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

# Copy from screen
$graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bounds.Size)

# Output based on format
if ($OutputFormat -eq "base64") {
    # Convert to base64 for easy transfer to WSL
    $stream = New-Object System.IO.MemoryStream
    $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
    $bytes = $stream.ToArray()
    [Convert]::ToBase64String($bytes)
    $stream.Dispose()
} else {
    # Save to file (for testing/debugging)
    $file = "$env:TEMP\screenshot_$(Get-Date -Format 'yyyyMMdd_HHmmss').png"
    $bitmap.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output $file
}

# Cleanup
$graphics.Dispose()
$bitmap.Dispose()