# SciTeX Metadata Embedding Demo

This demo showcases the image metadata embedding functionality in `scitex.io`.

## Overview

The metadata embedding feature allows you to attach research-related information directly to your image files (PNG and JPEG formats). This metadata travels with the image file, ensuring reproducibility and traceability of your scientific figures.

## Features

- **Automatic metadata embedding**: Add any JSON-serializable dictionary as metadata
- **Standard formats**: Uses PNG tEXt chunks and JPEG EXIF fields
- **Easy retrieval**: Read metadata back with a simple function call
- **Backward compatible**: Existing code works without changes
- **Visual metadata display**: Optionally display metadata at the bottom of images

## Installation

Ensure you have the required dependencies:

```bash
pip install pillow piexif matplotlib numpy
```

## Usage

### Basic Usage

```python
import scitex as stx
import matplotlib.pyplot as plt

# Create a plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])

# Save with metadata
metadata = {
    'experiment': 'test_001',
    'date': '2024-11-14',
    'subject': 'S001'
}
stx.io.save(fig, 'result.png', metadata=metadata)

# Read metadata
meta = stx.io.read_metadata('result.png')
print(meta['experiment'])  # 'test_001'
```

### Running the Demo

```bash
cd examples/metadata_demo
python demo_metadata_embedding.py
```

This will generate several example images in the `output/` directory:

1. **demo_basic.png**: Simple plot with embedded metadata (PNG format)
2. **demo_basic.jpg**: Same plot with metadata (JPEG format)
3. **demo_visual.png**: Plot with metadata visually displayed at the bottom
4. **demo_no_metadata.png**: Plot without metadata (backward compatibility test)

## Demo Components

### Demo 1: Basic Metadata Embedding
Shows how to:
- Create a plot with matplotlib
- Save it with metadata in both PNG and JPEG formats
- Read the metadata back

### Demo 2: Visual Metadata Display
Creates an image where metadata is:
- Embedded in the file (invisible to normal viewers)
- Visually displayed at the bottom of the image for human readers
- Separated by a "cut here" line

### Demo 3: Backward Compatibility
Verifies that:
- Images can still be saved without metadata
- The API remains unchanged for existing code

## Metadata Structure

You can use any JSON-serializable dictionary structure:

```python
metadata = {
    'experiment': 'seizure_prediction_001',
    'session': '2024-11-14_session_01',
    'analysis': 'PAC',
    'subject_id': 'S001',
    'electrode': 'Fp1',
    'sampling_rate': 1000,
    'created': '2024-11-14T10:30:00',
    'notes': 'Pre-ictal period recording'
}
```

## Use Cases

1. **Research reproducibility**: Track experiment parameters with each figure
2. **Lab notebooks**: Automatically document analysis settings
3. **Collaboration**: Share figures with embedded context
4. **Data provenance**: Maintain audit trail of analysis steps
5. **Automated workflows**: Generate self-documenting figures

## Technical Details

### PNG Format
- Uses standard PNG tEXt chunks
- Key: `scitex_metadata`
- Value: JSON string
- Readable by standard tools like `exiftool`

### JPEG Format
- Uses EXIF ImageDescription field
- Contains JSON string
- Compatible with photo management software

## Limitations

- Metadata is not visible in standard image viewers
- Use `stx.io.read_metadata()` or tools like `exiftool` to read
- JPEG compression is lossy (image quality), but metadata is preserved
- Large metadata (>10KB) may affect file size

## API Reference

### `stx.io.save(..., metadata=dict)`
Save an image with embedded metadata.

**Parameters:**
- `obj`: Figure or image object
- `path`: Output file path (PNG or JPEG)
- `metadata`: Dictionary of metadata (optional)

### `stx.io.read_metadata(path) -> dict`
Read metadata from an image file.

**Returns:**
- Dictionary containing metadata, or `None` if no metadata found

### `stx.io.has_metadata(path) -> bool`
Check if an image has embedded metadata.

**Returns:**
- `True` if metadata exists, `False` otherwise

## Philosophy

This implementation follows SciTeX's "user sovereignty" principle:
- Files remain standard format (no lock-in)
- Metadata travels with the file
- Users control what information to include
- Compatible with existing tools

## Future Enhancements

Potential additions discussed in the design memo:
- Auto-metadata collection (git info, environment, timestamps)
- CSV/database integration for searchable metadata
- QR codes for smartphone-readable metadata
- Batch metadata extraction tools
