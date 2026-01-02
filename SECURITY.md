# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of WaveDL seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:
- **Email**: ductho.le@outlook.com

Please include the following information in your report:
- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- You should receive an acknowledgment of your report within **48 hours**.
- We will keep you informed about the progress of fixing the vulnerability.
- We may ask for additional information or guidance during the investigation.
- Once the vulnerability is fixed, we will notify you and publicly disclose it (unless you prefer otherwise).

### Security Best Practices

When using WaveDL, please follow these security best practices:

1. **Keep dependencies up to date**: Regularly update PyTorch, NumPy, and other dependencies.
2. **Validate input data**: Always validate and sanitize data files before loading them.
3. **Use trusted data sources**: Only load .npz, .h5, .mat files from trusted sources, as they can contain arbitrary Python code when unpickled.
4. **Secure model checkpoints**: Treat model checkpoints as code - they can execute arbitrary operations during loading.
5. **Environment isolation**: Use virtual environments or containers to isolate WaveDL installations.
6. **Access control**: Restrict access to training data and model checkpoints containing sensitive information.

### Third-Party Dependencies

WaveDL depends on several third-party packages. We monitor their security advisories and update dependencies when security patches are released. Users should:

- Regularly run `pip install --upgrade wavedl` to get the latest secure versions
- Monitor PyTorch security advisories at https://github.com/pytorch/pytorch/security
- Review dependency security with tools like `pip-audit` or `safety`

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release new versions and publish security advisories

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue.
