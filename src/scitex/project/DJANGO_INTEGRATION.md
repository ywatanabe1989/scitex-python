# Django Integration Guide

This document explains how to integrate `scitex.project` with Django models in `scitex-cloud`.

## Overview

The integration uses an **adapter pattern** where:
1. **Core Logic** lives in `scitex.project` (pure Python, no Django)
2. **Django Models** wrap the core logic and add web-specific features
3. **One-to-One Mapping** is maintained between layers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Django Project Model (ORM)                    │
│  - id, owner (ForeignKey)                                    │
│  - Django-specific: created_at, updated_at                   │
│  - Integration: scitex_project_id, local_path                │
│  - Gitea: gitea_repo_id, gitea_repo_name, etc.              │
└─────────────────────────────────────────────────────────────┘
                              ▲ ▼
                    to_scitex_project() / sync_from_scitex()
                              ▲ ▼
┌─────────────────────────────────────────────────────────────┐
│              SciTeXProject (Dataclass)                       │
│  - Pure Python, no dependencies                              │
│  - Metadata in .scitex/ directory                            │
│  - Validation, storage, persistence                          │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Migration

### Phase 1: Add Integration Fields to Django Model

```python
# In apps/project_app/models.py

from django.db import models
from django.contrib.auth.models import User

class Project(models.Model):
    """Django Project model with SciTeX integration."""

    # Existing Django fields
    id = models.AutoField(primary_key=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='owned_projects')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Core fields (synced with SciTeXProject)
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField()
    visibility = models.CharField(max_length=20, choices=[('public', 'Public'), ('private', 'Private')], default='private')

    # NEW: Integration fields
    scitex_project_id = models.CharField(
        max_length=100,
        unique=True,
        null=True,
        blank=True,
        help_text="Unique identifier linking to .scitex/ metadata"
    )
    local_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="Path to local project directory (where .scitex/ lives)"
    )

    # Gitea integration (existing)
    gitea_repo_id = models.IntegerField(null=True, blank=True)
    gitea_repo_name = models.CharField(max_length=200, blank=True)
    gitea_enabled = models.BooleanField(default=False)

    # ... other fields ...
```

### Phase 2: Add Adapter Methods

```python
# In apps/project_app/models.py

from scitex.project import SciTeXProject, ProjectValidator, validate_name
from pathlib import Path

class Project(models.Model):
    # ... fields from above ...

    def get_local_path(self) -> Path:
        """
        Get Path object for local project directory.

        Returns:
            Path to project directory (e.g., data/users/ywatanabe/my-project/)
        """
        if not self.local_path:
            # Default location
            from apps.workspace_app.services.directory_service import get_user_directory_manager
            manager = get_user_directory_manager(self.owner)
            return manager.base_path / self.slug
        return Path(self.local_path)

    def has_scitex_metadata(self) -> bool:
        """Check if project has .scitex/ directory."""
        return (self.get_local_path() / '.scitex').exists()

    def to_scitex_project(self) -> SciTeXProject:
        """
        Convert Django model to SciTeXProject dataclass.

        Returns:
            SciTeXProject instance loaded from .scitex/ directory

        Raises:
            FileNotFoundError: If .scitex/ doesn't exist
        """
        if not self.has_scitex_metadata():
            raise FileNotFoundError(
                f"Project '{self.name}' has no .scitex/ metadata. "
                f"Call initialize_scitex_metadata() first."
            )

        return SciTeXProject.load_from_directory(self.get_local_path())

    def initialize_scitex_metadata(self) -> SciTeXProject:
        """
        Initialize .scitex/ directory for existing Django project.

        This is used during migration to add .scitex/ to projects that don't have it yet.

        Returns:
            Newly created SciTeXProject
        """
        if self.has_scitex_metadata():
            raise FileExistsError(f"Project '{self.name}' already has .scitex/ metadata")

        local_path = self.get_local_path()

        # Create directory if it doesn't exist
        local_path.mkdir(parents=True, exist_ok=True)

        # Create SciTeXProject
        scitex_project = SciTeXProject.create(
            name=self.name,
            path=local_path,
            owner=self.owner.username,
            description=self.description,
            visibility=self.visibility,
            template=None,  # Unknown for existing projects
            init_git=False  # Git might already be initialized
        )

        # Link Django project to SciTeXProject
        self.scitex_project_id = scitex_project.project_id
        self.local_path = str(local_path)
        self.save(update_fields=['scitex_project_id', 'local_path'])

        return scitex_project

    def sync_from_scitex(self) -> None:
        """
        Update Django model from SciTeXProject metadata.

        Use this when .scitex/ metadata is the source of truth (e.g., after local edits).
        """
        scitex_project = self.to_scitex_project()

        # Update fields from SciTeXProject
        self.name = scitex_project.name
        self.slug = scitex_project.slug
        self.description = scitex_project.description
        self.visibility = scitex_project.visibility
        self.storage_used = scitex_project.storage_used

        self.save()

    def sync_to_scitex(self) -> None:
        """
        Update SciTeXProject from Django model.

        Use this when Django model is the source of truth (e.g., after web UI changes).
        """
        scitex_project = self.to_scitex_project()

        # Update SciTeXProject fields
        scitex_project.name = self.name
        scitex_project.slug = self.slug
        scitex_project.description = self.description
        scitex_project.visibility = self.visibility

        # Save to .scitex/
        scitex_project.save()

    def update_storage_from_scitex(self) -> int:
        """
        Calculate storage using SciTeXProject and update Django model.

        Returns:
            Storage size in bytes
        """
        if not self.has_scitex_metadata():
            # Fall back to old method
            return self.update_storage_usage()

        scitex_project = self.to_scitex_project()
        storage = scitex_project.update_storage_usage()

        # Update Django model
        self.storage_used = storage
        self.save(update_fields=['storage_used'])

        return storage

    @classmethod
    def validate_name_using_scitex(cls, name: str):
        """
        Validate project name using scitex.project validator.

        Raises:
            ValidationError: If name is invalid
        """
        from django.core.exceptions import ValidationError

        is_valid, error = validate_name(name)
        if not is_valid:
            raise ValidationError(error)

    def clean(self):
        """Django model validation (called before save)."""
        super().clean()

        # Use scitex validator
        self.validate_name_using_scitex(self.name)

        # Generate slug if not set
        if not self.slug:
            from scitex.project import generate_slug
            self.slug = generate_slug(self.name)
```

### Phase 3: Update Views to Use SciTeX

```python
# In apps/project_app/views.py

from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from scitex.project import validate_name, generate_slug

@login_required
def create_project(request):
    """Create new project with SciTeX integration."""
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        visibility = request.POST.get('visibility', 'private')

        # Validate using scitex
        is_valid, error = validate_name(name)
        if not is_valid:
            return render(request, 'project_app/create.html', {
                'error': error
            })

        # Generate slug
        slug = generate_slug(name)

        # Check uniqueness (Django level)
        if Project.objects.filter(owner=request.user, name=name).exists():
            return render(request, 'project_app/create.html', {
                'error': f"You already have a project named '{name}'"
            })

        # Create Django project
        project = Project.objects.create(
            name=name,
            slug=slug,
            description=description,
            owner=request.user,
            visibility=visibility
        )

        # Initialize SciTeX metadata
        try:
            scitex_project = project.initialize_scitex_metadata()
            messages.success(request, f"Project '{name}' created successfully!")
        except Exception as e:
            # Rollback Django project if SciTeX init fails
            project.delete()
            return render(request, 'project_app/create.html', {
                'error': f"Failed to initialize project: {e}"
            })

        # Create Gitea repository if requested
        if request.POST.get('init_gitea'):
            try:
                project.create_gitea_repository()
            except Exception as e:
                messages.warning(request, f"Project created but Gitea init failed: {e}")

        return redirect('project_app:detail', username=request.user.username, slug=slug)

    return render(request, 'project_app/create.html')


@login_required
def project_storage(request, username, slug):
    """Show detailed storage breakdown."""
    project = get_object_or_404(Project, owner__username=username, slug=slug)

    if not project.can_view(request.user):
        return HttpResponseForbidden()

    # Use SciTeX for detailed breakdown
    if project.has_scitex_metadata():
        scitex_project = project.to_scitex_project()
        breakdown = scitex_project.get_storage_breakdown()
    else:
        # Fallback for projects without .scitex/
        breakdown = {'total': project.storage_used, 'breakdown': {}}

    return render(request, 'project_app/storage.html', {
        'project': project,
        'breakdown': breakdown
    })
```

## Migration Command

Create a management command to migrate existing projects:

```python
# In apps/project_app/management/commands/migrate_to_scitex.py

from django.core.management.base import BaseCommand
from apps.project_app.models import Project
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Migrate existing projects to use SciTeX metadata (.scitex/)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes'
        )
        parser.add_argument(
            '--username',
            type=str,
            help='Only migrate projects for specific user'
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        username = options.get('username')

        # Get projects to migrate
        projects = Project.objects.all()
        if username:
            projects = projects.filter(owner__username=username)

        total = projects.count()
        migrated = 0
        skipped = 0
        errors = 0

        self.stdout.write(f"Found {total} projects to process")

        for project in projects:
            # Skip if already has .scitex/
            if project.has_scitex_metadata():
                self.stdout.write(f"  SKIP: {project.owner.username}/{project.slug} (already has .scitex/)")
                skipped += 1
                continue

            # Skip if directory doesn't exist
            if not project.get_local_path().exists():
                self.stdout.write(
                    self.style.WARNING(
                        f"  SKIP: {project.owner.username}/{project.slug} (directory not found)"
                    )
                )
                skipped += 1
                continue

            if dry_run:
                self.stdout.write(f"  WOULD MIGRATE: {project.owner.username}/{project.slug}")
                migrated += 1
                continue

            # Migrate project
            try:
                scitex_project = project.initialize_scitex_metadata()
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ MIGRATED: {project.owner.username}/{project.slug} "
                        f"(ID: {scitex_project.project_id})"
                    )
                )
                migrated += 1

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(
                        f"  ✗ ERROR: {project.owner.username}/{project.slug} - {e}"
                    )
                )
                logger.error(f"Migration failed for project {project.id}: {e}", exc_info=True)
                errors += 1

        # Summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write(f"Total: {total}")
        self.stdout.write(self.style.SUCCESS(f"Migrated: {migrated}"))
        self.stdout.write(f"Skipped: {skipped}")
        if errors > 0:
            self.stdout.write(self.style.ERROR(f"Errors: {errors}"))

        if dry_run:
            self.stdout.write("\nThis was a dry run. Use without --dry-run to actually migrate.")
```

## Usage

```bash
# Migrate all projects (dry run)
python manage.py migrate_to_scitex --dry-run

# Migrate all projects (actual)
python manage.py migrate_to_scitex

# Migrate specific user's projects
python manage.py migrate_to_scitex --username ywatanabe

# In Django shell: test integration
python manage.py shell
>>> from apps.project_app.models import Project
>>> project = Project.objects.first()
>>> project.initialize_scitex_metadata()
>>> scitex = project.to_scitex_project()
>>> print(scitex)
>>> scitex.update_storage_usage()
```

## Benefits of This Approach

1. **Gradual Migration**: Add fields incrementally, no big-bang rewrite
2. **Backward Compatible**: Old projects work until migrated
3. **Pure Logic Extraction**: Complex validation/storage logic moves to scitex
4. **Testability**: Test core logic without Django
5. **Reusability**: Same logic in CLI, SDK, and Django
6. **Clear Separation**: Django = web layer, scitex = business logic

## Testing Strategy

```python
# In apps/project_app/tests.py

from django.test import TestCase
from django.contrib.auth.models import User
from apps.project_app.models import Project
from pathlib import Path
import tempfile
import shutil


class ProjectSciTeXIntegrationTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='test')
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialize_scitex_metadata(self):
        """Test creating .scitex/ for existing project."""
        project = Project.objects.create(
            name="Test Project",
            slug="test-project",
            owner=self.user,
            local_path=str(self.temp_dir / "test-project")
        )

        # Initialize SciTeX metadata
        scitex_project = project.initialize_scitex_metadata()

        # Check .scitex/ exists
        self.assertTrue(project.has_scitex_metadata())
        self.assertEqual(scitex_project.name, "Test Project")
        self.assertEqual(scitex_project.owner, "testuser")

    def test_sync_from_scitex(self):
        """Test syncing Django model from .scitex/ changes."""
        project = Project.objects.create(
            name="Original Name",
            slug="original-name",
            owner=self.user,
            local_path=str(self.temp_dir / "test-project")
        )

        scitex_project = project.initialize_scitex_metadata()

        # Change in .scitex/
        scitex_project.description = "Updated description"
        scitex_project.save()

        # Sync to Django
        project.sync_from_scitex()

        # Verify Django model updated
        project.refresh_from_db()
        self.assertEqual(project.description, "Updated description")

    def test_validate_name_using_scitex(self):
        """Test name validation using scitex validator."""
        from django.core.exceptions import ValidationError

        # Valid name
        try:
            Project.validate_name_using_scitex("valid-project-name")
        except ValidationError:
            self.fail("Valid name raised ValidationError")

        # Invalid name (spaces)
        with self.assertRaises(ValidationError):
            Project.validate_name_using_scitex("invalid project name")

        # Invalid name (starts with hyphen)
        with self.assertRaises(ValidationError):
            Project.validate_name_using_scitex("-invalid")
```

## Next Steps

1. **Add Integration Fields**: Update Django model with `scitex_project_id` and `local_path`
2. **Add Adapter Methods**: Implement `to_scitex_project()`, `sync_from_scitex()`, etc.
3. **Create Migration Command**: `migrate_to_scitex.py`
4. **Update Views**: Use scitex validators and storage calculators
5. **Test Integration**: Write tests for Django ↔ SciTeX sync
6. **Migrate Projects**: Run migration command on existing projects
