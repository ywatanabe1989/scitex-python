# Django Integration Guide for scitex.vis

**Integration with `/vis/sigma/` Django Application**

---

## Overview

This guide explains how to integrate `scitex.vis` with the Django-based `/vis/sigma/` application for web-based figure editing and management.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Frontend (React/Vue)                       │
│  - Tree view of figure structure            │
│  - Canvas for visual editing                │
│  - Form controls for plot parameters        │
└────────────┬────────────────────────────────┘
             │ HTTP/REST API
┌────────────▼────────────────────────────────┐
│  Django Backend (/vis/sigma/)               │
│  - Figure CRUD endpoints                    │
│  - Export endpoints                         │
│  - Project management                       │
└────────────┬────────────────────────────────┘
             │ Python API
┌────────────▼────────────────────────────────┐
│  scitex.vis                                 │
│  - JSON validation                          │
│  - Figure rendering                         │
│  - Export to PNG/PDF/SVG                    │
└─────────────────────────────────────────────┘
```

---

## Required Django API Endpoints

### 1. Figure Management

#### List Figures in Project

```python
# GET /api/vis/figures/
def list_figures(request):
    """List all figures in the project."""
    import scitex as stx

    project_dir = get_project_dir(request)  # Your implementation

    figure_ids = stx.vis.io.list_figures_in_project(project_dir)

    return JsonResponse({
        'figures': [
            {
                'id': fig_id,
                'path': f'{project_dir}/scitex/vis/figs/{fig_id}.json'
            }
            for fig_id in figure_ids
        ]
    })
```

#### Get Figure JSON

```python
# GET /api/vis/figures/{figure_id}/
def get_figure(request, figure_id):
    """Get figure JSON by ID."""
    import scitex as stx

    project_dir = get_project_dir(request)

    try:
        fig_json = stx.vis.load_figure_json_from_project(
            project_dir=project_dir,
            figure_id=figure_id
        )
        return JsonResponse(fig_json)

    except FileNotFoundError:
        return JsonResponse(
            {'error': f'Figure {figure_id} not found'},
            status=404
        )
```

#### Create/Update Figure

```python
# POST /api/vis/figures/{figure_id}/
def save_figure(request, figure_id):
    """Save or update figure JSON."""
    import scitex as stx
    from scitex.vis.backend import validate_figure_json

    project_dir = get_project_dir(request)
    fig_json = json.loads(request.body)

    # Validate before saving
    try:
        validate_figure_json(fig_json)
    except ValueError as e:
        return JsonResponse(
            {'error': f'Invalid figure JSON: {e}'},
            status=400
        )

    # Save to project
    path = stx.vis.save_figure_json_to_project(
        project_dir=project_dir,
        figure_id=figure_id,
        fig_json=fig_json
    )

    return JsonResponse({
        'success': True,
        'figure_id': figure_id,
        'path': str(path)
    })
```

#### Delete Figure

```python
# DELETE /api/vis/figures/{figure_id}/
def delete_figure(request, figure_id):
    """Delete figure JSON."""
    project_dir = get_project_dir(request)
    json_path = Path(project_dir) / 'scitex' / 'vis' / 'figs' / f'{figure_id}.json'

    if json_path.exists():
        json_path.unlink()
        return JsonResponse({'success': True})
    else:
        return JsonResponse(
            {'error': f'Figure {figure_id} not found'},
            status=404
        )
```

### 2. Export Endpoints

#### Export Figure to Image

```python
# GET /api/vis/figures/{figure_id}/export?format=png&dpi=300
def export_figure(request, figure_id):
    """Export figure to image format."""
    import scitex as stx

    project_dir = get_project_dir(request)

    # Get parameters
    fmt = request.GET.get('format', 'png')  # png, pdf, svg
    dpi = int(request.GET.get('dpi', 300))
    auto_crop = request.GET.get('auto_crop', 'false').lower() == 'true'

    # Load figure JSON
    try:
        fig_json = stx.vis.load_figure_json_from_project(
            project_dir=project_dir,
            figure_id=figure_id
        )
    except FileNotFoundError:
        return JsonResponse(
            {'error': f'Figure {figure_id} not found'},
            status=404
        )

    # Export to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(
        suffix=f'.{fmt}',
        delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        stx.vis.export_figure(
            fig_json=fig_json,
            output_path=tmp_path,
            fmt=fmt,
            dpi=dpi,
            auto_crop=auto_crop
        )

        # Return file
        with open(tmp_path, 'rb') as f:
            response = HttpResponse(
                f.read(),
                content_type=f'image/{fmt}' if fmt in ['png', 'svg'] else 'application/pdf'
            )
            response['Content-Disposition'] = f'attachment; filename="{figure_id}.{fmt}"'
            return response

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

#### Export Multiple Formats

```python
# POST /api/vis/figures/{figure_id}/export-multi/
def export_multiple_formats(request, figure_id):
    """Export figure to multiple formats simultaneously."""
    import scitex as stx

    project_dir = get_project_dir(request)
    formats = request.POST.getlist('formats', ['png', 'pdf', 'svg'])

    # Load figure
    fig_json = stx.vis.load_figure_json_from_project(
        project_dir=project_dir,
        figure_id=figure_id
    )

    # Export to project export directory
    export_dir = Path(project_dir) / 'scitex' / 'vis' / 'export'

    paths = stx.vis.backend.export_multiple_formats(
        fig_json=fig_json,
        output_dir=export_dir,
        base_name=figure_id,
        formats=formats,
        dpi=300,
        auto_crop=True
    )

    return JsonResponse({
        'success': True,
        'files': {fmt: str(path) for fmt, path in paths.items()}
    })
```

### 3. Template Endpoints

#### List Available Templates

```python
# GET /api/vis/templates/
def list_templates(request):
    """List available figure templates."""
    import scitex as stx

    templates = stx.vis.list_templates()

    template_info = []
    for name in templates:
        template = stx.vis.get_template(name)
        template_info.append({
            'name': name,
            'width_mm': template['width_mm'],
            'height_mm': template['height_mm'],
            'description': template.get('metadata', {}).get('template', name)
        })

    return JsonResponse({'templates': template_info})
```

#### Get Template

```python
# GET /api/vis/templates/{template_name}/
def get_template(request, template_name):
    """Get a specific template."""
    import scitex as stx

    try:
        # Get optional parameters
        height_mm = request.GET.get('height_mm')
        nrows = request.GET.get('nrows')
        ncols = request.GET.get('ncols')

        kwargs = {}
        if height_mm:
            kwargs['height_mm'] = float(height_mm)
        if nrows:
            kwargs['nrows'] = int(nrows)
        if ncols:
            kwargs['ncols'] = int(ncols)

        template = stx.vis.get_template(template_name, **kwargs)
        return JsonResponse(template)

    except ValueError as e:
        return JsonResponse(
            {'error': f'Unknown template: {template_name}'},
            status=404
        )
```

### 4. Validation Endpoint

```python
# POST /api/vis/validate/
def validate_figure_json(request):
    """Validate figure JSON without saving."""
    from scitex.vis.backend import validate_figure_json

    fig_json = json.loads(request.body)

    try:
        validate_figure_json(fig_json)
        return JsonResponse({
            'valid': True,
            'message': 'Figure JSON is valid'
        })

    except ValueError as e:
        return JsonResponse({
            'valid': False,
            'error': str(e)
        }, status=400)
```

---

## URL Configuration

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Figure management
    path('api/vis/figures/', views.list_figures, name='list_figures'),
    path('api/vis/figures/<str:figure_id>/', views.get_figure, name='get_figure'),
    path('api/vis/figures/<str:figure_id>/', views.save_figure, name='save_figure'),
    path('api/vis/figures/<str:figure_id>/', views.delete_figure, name='delete_figure'),

    # Export
    path('api/vis/figures/<str:figure_id>/export/', views.export_figure, name='export_figure'),
    path('api/vis/figures/<str:figure_id>/export-multi/', views.export_multiple_formats, name='export_multiple'),

    # Templates
    path('api/vis/templates/', views.list_templates, name='list_templates'),
    path('api/vis/templates/<str:template_name>/', views.get_template, name='get_template'),

    # Validation
    path('api/vis/validate/', views.validate_figure_json, name='validate'),
]
```

---

## Frontend Integration

### Loading CSV Data for Plots

When the frontend needs to create a plot, it should:

1. **Upload CSV** to Django backend
2. **Parse CSV** to extract x, y data
3. **Convert to JSON arrays** for plot data
4. **Send to scitex.vis** as part of figure JSON

```python
# POST /api/vis/data/upload/
def upload_plot_data(request):
    """Upload CSV and convert to plot data."""
    import pandas as pd
    import io

    csv_file = request.FILES['file']
    x_column = request.POST.get('x_column', 0)
    y_column = request.POST.get('y_column', 1)

    # Read CSV
    df = pd.read_csv(io.BytesIO(csv_file.read()))

    # Extract columns
    x_data = df.iloc[:, x_column].tolist()
    y_data = df.iloc[:, y_column].tolist()

    return JsonResponse({
        'data': {
            'x': x_data,
            'y': y_data
        },
        'columns': df.columns.tolist()
    })
```

### Frontend Workflow

1. **Select Template**
   ```javascript
   const template = await fetch('/api/vis/templates/nature_single/').then(r => r.json());
   ```

2. **Upload Data**
   ```javascript
   const formData = new FormData();
   formData.append('file', csvFile);
   formData.append('x_column', 0);
   formData.append('y_column', 1);

   const plotData = await fetch('/api/vis/data/upload/', {
     method: 'POST',
     body: formData
   }).then(r => r.json());
   ```

3. **Build Figure JSON**
   ```javascript
   const figJson = {
     ...template,
     axes: [{
       row: 0,
       col: 0,
       xlabel: "Time (s)",
       ylabel: "Amplitude",
       plots: [{
         plot_type: "line",
         data: plotData.data,
         color: "blue",
         linewidth: 2
       }]
     }]
   };
   ```

4. **Save Figure**
   ```javascript
   await fetch('/api/vis/figures/fig-001/', {
     method: 'POST',
     headers: {'Content-Type': 'application/json'},
     body: JSON.stringify(figJson)
   });
   ```

5. **Export**
   ```javascript
   const imageBlob = await fetch(
     '/api/vis/figures/fig-001/export?format=png&dpi=300'
   ).then(r => r.blob());
   ```

---

## Project Directory Structure

```
project/
├── scitex/
│   └── vis/
│       ├── figs/           # Figure JSON specifications
│       │   ├── fig-001.json
│       │   ├── fig-002.json
│       │   └── ...
│       ├── export/         # Exported images
│       │   ├── fig-001.png
│       │   ├── fig-001.pdf
│       │   └── ...
│       └── data/           # Optional: CSV data files
│           ├── dataset-01.csv
│           └── ...
└── ...
```

---

## Security Considerations

### 1. Input Validation

Always validate figure JSON before processing:

```python
from scitex.vis.backend import validate_figure_json

try:
    validate_figure_json(fig_json)
except ValueError as e:
    return JsonResponse({'error': str(e)}, status=400)
```

### 2. File Path Sanitization

Prevent directory traversal attacks:

```python
import os
from pathlib import Path

def get_safe_figure_path(project_dir, figure_id):
    """Get sanitized figure path."""
    # Remove any path components
    safe_id = Path(figure_id).name

    # Ensure .json extension
    if not safe_id.endswith('.json'):
        safe_id = f'{safe_id}.json'

    path = Path(project_dir) / 'scitex' / 'vis' / 'figs' / safe_id

    # Ensure path is within project directory
    if not str(path.resolve()).startswith(str(Path(project_dir).resolve())):
        raise ValueError('Invalid figure ID')

    return path
```

### 3. Rate Limiting

Apply rate limiting to export endpoints:

```python
from django.views.decorators.cache import cache_page
from django.views.decorators.ratelimit import ratelimit

@ratelimit(key='user', rate='10/m', method='GET')
def export_figure(request, figure_id):
    # ... export logic
```

---

## Performance Optimization

### 1. Caching

Cache rendered figures:

```python
from django.core.cache import cache

def export_figure(request, figure_id):
    cache_key = f'figure_{figure_id}_{fmt}_{dpi}'

    cached = cache.get(cache_key)
    if cached:
        return cached

    # Render and cache
    response = render_and_export(fig_json, fmt, dpi)
    cache.set(cache_key, response, timeout=3600)  # 1 hour
    return response
```

### 2. Background Tasks

Use Celery for slow export operations:

```python
from celery import shared_task

@shared_task
def export_figure_task(project_dir, figure_id, fmt, dpi):
    """Background task for figure export."""
    import scitex as stx

    fig_json = stx.vis.load_figure_json_from_project(project_dir, figure_id)
    output_path = Path(project_dir) / 'scitex' / 'vis' / 'export' / f'{figure_id}.{fmt}'

    stx.vis.export_figure(fig_json, output_path, fmt=fmt, dpi=dpi)

    return str(output_path)

# In view:
def export_figure(request, figure_id):
    task = export_figure_task.delay(project_dir, figure_id, fmt, dpi)
    return JsonResponse({'task_id': task.id})
```

---

## Testing

### Unit Tests

```python
from django.test import TestCase
import scitex as stx

class VisFigureTests(TestCase):
    def test_save_and_load_figure(self):
        """Test figure save/load cycle."""
        fig_json = stx.vis.get_template('square')

        # Save
        path = stx.vis.save_figure_json_to_project(
            '/tmp/test_project',
            'test-fig',
            fig_json
        )

        # Load
        loaded = stx.vis.load_figure_json_from_project(
            '/tmp/test_project',
            'test-fig'
        )

        self.assertEqual(fig_json['width_mm'], loaded['width_mm'])

    def test_export_figure(self):
        """Test figure export."""
        fig_json = stx.vis.get_template('square')
        fig_json['axes'] = [{
            'plots': [{
                'plot_type': 'line',
                'data': {'x': [0, 1, 2], 'y': [0, 1, 4]}
            }]
        }]

        output = '/tmp/test.png'
        stx.vis.export_figure(fig_json, output, dpi=150)

        self.assertTrue(Path(output).exists())
```

### Integration Tests

```python
class VisAPITests(TestCase):
    def test_list_figures(self):
        """Test figure listing endpoint."""
        response = self.client.get('/api/vis/figures/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('figures', response.json())

    def test_save_figure(self):
        """Test figure save endpoint."""
        fig_json = stx.vis.get_template('square')

        response = self.client.post(
            '/api/vis/figures/test-fig/',
            data=json.dumps(fig_json),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
```

---

## Troubleshooting

### Common Issues

1. **Figure not rendering**
   - Check if scitex.plt is properly installed
   - Verify matplotlib backend is available
   - Check for missing data in plot configurations

2. **Validation errors**
   - Use `/api/vis/validate/` endpoint to check JSON
   - Review error messages for specific field issues
   - Ensure all required fields are present

3. **Export failures**
   - Check disk space for export directory
   - Verify write permissions
   - Check matplotlib/scitex.plt configuration

---

## Next Steps

1. **Implement Django views** using the examples above
2. **Create URL patterns** for all endpoints
3. **Build frontend components** for:
   - Figure tree viewer
   - Visual canvas editor
   - Form controls for plot parameters
4. **Test integration** with sample figures
5. **Deploy** to production environment

---

**Integration complete! Ready to connect `/vis/sigma/` Django app with `scitex.vis`** 🚀
