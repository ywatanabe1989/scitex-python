# Project Naming: Name vs Slug

This document explains the distinction between `name` and `slug` in SciTeX projects.

## TL;DR

- **Standalone mode**: `name` = `slug` (simple, no database)
- **Django/Cloud mode**: `name` is local, `slug` is globally unique URL identifier

## Detailed Explanation

### Standalone Mode (scitex.project only)

When using `scitex.project` without Django:

```python
from scitex.project import SciTeXProject
from pathlib import Path

project = SciTeXProject.create(
    name="neural-decoding",
    path=Path("/home/user/projects/neural-decoding"),
    owner="ywatanabe"
)

print(project.name)  # "neural-decoding"
print(project.slug)  # "neural-decoding" (same as name)
```

**In standalone mode:**
- `name` = your project identifier
- `slug` = automatically generated from `name` (usually identical)
- No global uniqueness enforcement (no database to check)
- Perfect for local work, containers, personal projects

### Django/Cloud Mode

When integrated with scitex-cloud Django:

```python
# User ywatanabe creates project
project1 = Project.objects.create(
    name="neural-decoding",
    owner=ywatanabe
)
# Django auto-generates: slug="neural-decoding"
# URL: https://scitex.cloud/ywatanabe/neural-decoding/
# Gitea: https://gitea.scitex.cloud/ywatanabe/neural-decoding

# User tanaka also creates project with same name
project2 = Project.objects.create(
    name="neural-decoding",
    owner=tanaka
)
# Django auto-generates: slug="neural-decoding-1" (globally unique!)
# URL: https://scitex.cloud/tanaka/neural-decoding-1/
# Gitea: https://gitea.scitex.cloud/tanaka/neural-decoding-1
```

**In Django mode:**
- `name` = user's chosen name (unique per user)
- `slug` = **globally unique** URL identifier (managed by Django)
- If multiple users use same name, Django adds `-1`, `-2`, etc. to slug
- Slug is used in URLs, Gitea repo names, directory names

## Why This Design?

### GitHub-Style URLs

Like GitHub, we use URLs of the form:
```
https://scitex.cloud/{username}/{slug}/
```

This requires `slug` to be globally unique across all users (you can't have two URLs with the same path).

### Real-World Example

**GitHub:**
- User `facebook` creates repo: `https://github.com/facebook/react`
- User `personal-dev` can also name their repo "react": `https://github.com/personal-dev/react`
- Same name, different users, no conflict because `{username}/{repo-name}` is unique

**SciTeX (same pattern):**
- User `ywatanabe` creates: `https://scitex.cloud/ywatanabe/neural-decoding`
- User `tanaka` creates: `https://scitex.cloud/tanaka/neural-decoding`
- No conflict! Username provides namespace separation

**But wait, why do we need globally unique slugs then?**

Because in our current Django implementation (line 54 of models.py), `slug` is `unique=True` globally. This means if two users both want project name "neural-decoding", Django will:
1. First user: gets slug `"neural-decoding"`
2. Second user: gets slug `"neural-decoding-1"` (to maintain global uniqueness)

### Alternative Design (GitHub-style)

We could change Django to use `unique_together = ('owner', 'slug')` instead of global slug uniqueness. Then:

```python
# models.py
class Project(models.Model):
    slug = models.SlugField(max_length=200)  # Remove unique=True

    class Meta:
        unique_together = [
            ('name', 'owner'),  # Already exists
            ('slug', 'owner'),  # Add this
        ]
```

Then:
- User `ywatanabe` → name=`"neural-decoding"` → slug=`"neural-decoding"`
- User `tanaka` → name=`"neural-decoding"` → slug=`"neural-decoding"` ✅ (allowed!)
- URLs: `/ywatanabe/neural-decoding/` and `/tanaka/neural-decoding/` (no conflict)

## Current Implementation

### In scitex.project (Standalone)

```python
# src/scitex/project/core.py

@dataclass
class SciTeXProject:
    name: str        # Project name
    slug: str = ""   # Auto-generated from name (defaults to name)

    def __post_init__(self):
        if not self.slug:
            self.slug = generate_slug(self.name)
```

### In Django (scitex-cloud)

```python
# apps/project_app/models.py

class Project(models.Model):
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)  # Globally unique!
    owner = models.ForeignKey(User, ...)

    class Meta:
        unique_together = ('name', 'owner')  # Name unique per user

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = self.generate_unique_slug(self.name)  # Adds -1, -2 if needed
        super().save(*args, **kwargs)
```

## Recommendations

### For Standalone Users

Keep it simple:
```python
# Project name = directory name = slug
project = SciTeXProject.create(
    name="my-project",
    path=Path("/path/to/my-project"),
    owner="me"
)
# slug will be auto-generated as "my-project"
```

### For Django Integration

Django handles slug uniqueness automatically:
```python
# In Django view
from scitex.project import validate_name, generate_slug

name = request.POST.get('name')
is_valid, error = validate_name(name)  # Use scitex validator

if is_valid:
    project = Project.objects.create(
        name=name,
        owner=request.user
        # slug auto-generated by Django's save()
    )

    # Initialize scitex metadata
    scitex_project = project.initialize_scitex_metadata()
    # scitex_project.name = name
    # scitex_project.slug = Django's generated slug
```

## Common Questions

### Q: Why not just use `name` everywhere?

**A:** URLs require globally unique identifiers. If two users both name their project "test", we need different URLs:
- `/user1/test/` ✅
- `/user2/test/` ✅

But if slug must be globally unique:
- User1: slug=`"test"`
- User2: slug=`"test-1"` (auto-numbered)

### Q: Should I change Django to use `(owner, slug)` uniqueness instead?

**A:** Yes, this would be more GitHub-like! Then:
- Multiple users can have same slug
- URLs still work: `/{username}/{slug}/`
- Simpler logic (no auto-numbering needed)

**Migration:**
```python
# In migration
class Migration(migrations.Migration):
    operations = [
        migrations.AlterField(
            model_name='project',
            name='slug',
            field=models.SlugField(max_length=200),  # Remove unique=True
        ),
        migrations.AlterUniqueTogether(
            name='project',
            unique_together={('name', 'owner'), ('slug', 'owner')},
        ),
    ]
```

### Q: What about Gitea repository names?

**A:** Gitea already uses `{username}/{repo-name}` which is naturally namespaced:
```python
# Already correct in code (line 405, 443 of models.py)
client.get_repository(self.owner.username, self.slug)
# → ywatanabe/neural-decoding
# → tanaka/neural-decoding
# No conflict! Different usernames provide namespace separation
```

## Conclusion

**Current implementation:**
- Slug is globally unique (with auto-numbering)
- Works but not ideal

**Recommended implementation:**
- Change Django to `unique_together = ('slug', 'owner')`
- More GitHub-like
- Simpler (no auto-numbering)
- URLs still unique: `/{username}/{slug}/`

**For scitex.project standalone:**
- Keep `name` = `slug` (simple)
- No changes needed
