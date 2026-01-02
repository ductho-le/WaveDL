# Repository Configuration Guide

This document provides guidance on configuring the WaveDL GitHub repository for optimal community engagement and project management.

## GitHub Repository Topics

Recommended topics to add to the repository (via Settings > General > Topics):

```
deep-learning
pytorch
wave-propagation
inverse-problems
ultrasonic-testing
non-destructive-testing
guided-waves
material-characterization
structural-health-monitoring
geophysics
seismology
biomedical-imaging
elastography
machine-learning
regression
multi-gpu
distributed-training
hpc
scientific-computing
research
ultrasound
```

These topics help users discover WaveDL through GitHub's topic-based search.

## Branch Protection Rules

Recommended settings for the `main` branch (Settings > Branches > Add rule):

### Branch name pattern
```
main
```

### Protection rules
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: 1
  - ✅ Dismiss stale pull request approvals when new commits are pushed
  - ✅ Require review from Code Owners (if CODEOWNERS file exists)

- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - Required status checks:
    - `test` (from test.yml workflow)
    - `lint` (from lint.yml workflow)
    - `CodeQL` (from codeql.yml workflow)

- ✅ **Require conversation resolution before merging**

- ✅ **Require signed commits** (optional, for enhanced security)

- ✅ **Require linear history** (optional, for clean git history)

- ✅ **Include administrators** (apply rules to admins too)

- ❌ **Allow force pushes** (disabled for safety)

- ❌ **Allow deletions** (disabled for safety)

## Repository Settings

### General Settings

**Features:**
- ✅ Issues
- ✅ Discussions (already enabled)
- ✅ Projects (optional)
- ✅ Wiki (optional, but README is comprehensive)
- ✅ Sponsorships (if FUNDING.yml is configured)

**Pull Requests:**
- ✅ Allow squash merging (recommended)
- ✅ Allow merge commits
- ✅ Allow rebase merging
- ✅ Automatically delete head branches

**Archives:**
- ❌ Archive this repository (keep active)

### Security Settings

**Security & analysis:**
- ✅ **Dependency graph** - Enabled (tracks dependencies)
- ✅ **Dependabot alerts** - Enabled (notifies of vulnerabilities)
- ✅ **Dependabot security updates** - Enabled (auto-creates PRs for security fixes)
- ✅ **Dependabot version updates** - Enabled via `.github/dependabot.yml`
- ✅ **Code scanning** - Enabled via CodeQL workflow
- ✅ **Secret scanning** - Enabled (detects leaked secrets)
- ✅ **Secret scanning push protection** - Enabled (prevents secret commits)

### Notifications

**Watching:**
- Repository admins should watch the repository for:
  - ✅ All activity
  - ✅ Issues
  - ✅ Pull requests
  - ✅ Discussions
  - ✅ Security alerts

### Actions Settings

**Actions permissions:**
- ✅ Allow all actions and reusable workflows (or allow specific actions)
- ✅ Allow GitHub Actions to create and approve pull requests (for Dependabot)

**Workflow permissions:**
- ✅ Read and write permissions (needed for automated releases)
- ✅ Allow GitHub Actions to create and approve pull requests

### Pages (Optional)

If you want to publish documentation:
- **Source**: Deploy from a branch or GitHub Actions
- **Branch**: `gh-pages` (or create documentation site)
- **Custom domain**: Optional

## Issue and PR Labels

Recommended labels for better organization:

### Type Labels
- `bug` - Something isn't working (red: #d73a4a)
- `enhancement` - New feature or request (blue: #a2eeef)
- `documentation` - Improvements or additions to documentation (blue: #0075ca)
- `question` - Further information is requested (pink: #d876e3)

### Priority Labels
- `priority: critical` - Critical issue, immediate attention needed (red: #b60205)
- `priority: high` - High priority (orange: #ff9800)
- `priority: medium` - Medium priority (yellow: #ffeb3b)
- `priority: low` - Low priority (green: #4caf50)

### Status Labels
- `status: triage` - Needs initial review (gray: #ededed)
- `status: in-progress` - Work in progress (yellow: #fbca04)
- `status: blocked` - Blocked by dependencies (red: #e11d21)
- `status: help-wanted` - Extra attention is needed (green: #008672)
- `status: good-first-issue` - Good for newcomers (purple: #7057ff)

### Component Labels
- `models` - Related to model architectures
- `data-loading` - Related to data pipeline
- `training` - Related to training loop
- `inference` - Related to testing/inference
- `ci-cd` - Related to CI/CD pipelines
- `dependencies` - Related to dependencies (auto from Dependabot)
- `github-actions` - Related to GitHub Actions (auto from Dependabot)

### Special Labels
- `duplicate` - This issue or PR already exists
- `invalid` - This doesn't seem right
- `wontfix` - This will not be worked on
- `stale` - Auto-applied by stale bot
- `never-stale` - Exempt from stale bot
- `security` - Security-related issue
- `roadmap` - Future planned features

## CODEOWNERS File (Optional)

Create `.github/CODEOWNERS` to automatically request reviews:

```
# Default owner for everything
* @ductho-le

# Models
/models/ @ductho-le

# Utils
/utils/ @ductho-le

# Documentation
*.md @ductho-le
/docs/ @ductho-le

# CI/CD
/.github/ @ductho-le
```

## README Badges

Current badges are comprehensive. Optional additions:

```markdown
[![GitHub Stars](https://img.shields.io/github/stars/ductho-le/WaveDL?style=plastic&logo=github)](https://github.com/ductho-le/WaveDL/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ductho-le/WaveDL?style=plastic&logo=github)](https://github.com/ductho-le/WaveDL/network/members)
[![Contributors](https://img.shields.io/github/contributors/ductho-le/WaveDL?style=plastic&logo=github)](https://github.com/ductho-le/WaveDL/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/ductho-le/WaveDL?style=plastic&logo=github)](https://github.com/ductho-le/WaveDL/commits/main)
```

## Social Preview

Upload a social preview image (Settings > General > Social preview):
- Recommended size: 1280x640 pixels
- Can use the WaveDL logo or a representative image
- Appears when repository is shared on social media

## About Section

Repository description (appears under the repo name):
```
A Scalable Deep Learning Framework for Wave-Based Inverse Problems - Production-ready multi-GPU training for ultrasonic NDE, geophysics, and biomedical applications
```

Website (optional):
```
https://doi.org/10.5281/zenodo.18012338
```

## Environment Variables and Secrets

For CI/CD and automation, no additional secrets are needed currently.
GitHub automatically provides `GITHUB_TOKEN` for workflows.

If adding deployment or publishing:
- `PYPI_TOKEN` - For publishing to PyPI
- `CODECOV_TOKEN` - For code coverage reporting (if using Codecov)

---

**Note:** Many of these settings require repository admin access.
This document serves as a guide for repository maintainers.
