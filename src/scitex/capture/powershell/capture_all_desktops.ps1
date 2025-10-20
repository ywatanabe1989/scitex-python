# Capture screenshots from all virtual desktops in Windows 10/11
# This uses Windows.Forms to capture each monitor (virtual desktops typically span monitors)

param(
    [Parameter(Mandatory=$false)]
    [string]$OutputFormat = "base64"  # "base64" or "file"
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Enable DPI awareness for high-resolution capture
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class User32 {
    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
}
'@
$null = [User32]::SetProcessDPIAware()

# Get all screens (monitors)
$screens = [System.Windows.Forms.Screen]::AllScreens

$results = @()

foreach ($screen in $screens) {
    try {
        # Create bitmap for this screen
        $bounds = $screen.Bounds
        $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)

        # Set high quality rendering
        $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

        # Capture from screen
        $graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bitmap.Size)

        # Convert to base64
        $stream = New-Object System.IO.MemoryStream
        $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
        $bytes = $stream.ToArray()
        $base64 = [Convert]::ToBase64String($bytes)

        $results += @{
            DeviceName = $screen.DeviceName
            IsPrimary = $screen.Primary
            Bounds = @{
                X = $bounds.X
                Y = $bounds.Y
                Width = $bounds.Width
                Height = $bounds.Height
            }
            Base64Data = $base64
        }

        # Cleanup
        $graphics.Dispose()
        $bitmap.Dispose()
        $stream.Dispose()
    }
    catch {
        Write-Error "Failed to capture screen $($screen.DeviceName): $_"
    }
}

# Output as JSON
$output = @{
    TotalScreens = $results.Count
    Screens = $results
    Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}

$output | ConvertTo-Json -Depth 4
