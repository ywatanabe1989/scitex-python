param(
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
}
'@

# Set process DPI aware
$null = [User32]::SetProcessDPIAware()

# Get virtual screen (all monitors combined)
$virtualScreen = [System.Windows.Forms.SystemInformation]::VirtualScreen

# Create bitmap for entire virtual screen
$bitmap = New-Object System.Drawing.Bitmap $virtualScreen.Width, $virtualScreen.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)

# Set high quality rendering
$graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
$graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

# Copy from entire virtual screen
$graphics.CopyFromScreen($virtualScreen.X, $virtualScreen.Y, 0, 0, $virtualScreen.Size)

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
    $file = "$env:TEMP\screenshot_all_$(Get-Date -Format 'yyyyMMdd_HHmmss').png"
    $bitmap.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
    
    # Get monitor info for display
    $screens = [System.Windows.Forms.Screen]::AllScreens
    Write-Host "Captured $($screens.Count) monitor(s):"
    foreach ($screen in $screens) {
        $bounds = $screen.Bounds
        Write-Host "  - $($screen.DeviceName): $($bounds.Width)x$($bounds.Height) at ($($bounds.X), $($bounds.Y))"
    }
    Write-Output $file
}

# Cleanup
$graphics.Dispose()
$bitmap.Dispose()