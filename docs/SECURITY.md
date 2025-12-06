# Security Policy

## Supported Versions

Atlas is currently in active development. Security updates will be applied to the latest version on the `main` branch.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Atlas seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do Not** Open a Public Issue

Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

### 2. Report Privately

Send a detailed report to the project maintainers via:
- GitHub Security Advisories (preferred)
- Direct message to the repository owner

### 3. Include Details

Your report should include:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)
- Your contact information

### 4. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-14 days
  - Medium: 14-30 days
  - Low: 30-90 days

## Security Best Practices

When using Atlas:

### Training Security

1. **Data Privacy**: Be cautious with training data containing sensitive information
2. **Checkpoint Security**: Store model checkpoints securely with appropriate file permissions
3. **Environment Isolation**: Use virtual environments to isolate dependencies

### Inference Security

1. **Input Validation**: Sanitize user inputs before feeding to the model
2. **Output Filtering**: Review generated outputs before displaying to end users
3. **Resource Limits**: Set appropriate `max_new_tokens` limits to prevent resource exhaustion

### Dependencies

1. **Regular Updates**: Keep dependencies up to date
2. **Audit Packages**: Review `requirements.txt` for known vulnerabilities
3. **Use Official Sources**: Install packages from trusted sources (PyPI)

### Deployment Security

1. **API Security**: If deploying as a service, implement authentication and rate limiting
2. **HTTPS Only**: Use encrypted connections for model serving
3. **Monitoring**: Log and monitor for unusual activity or abuse

## Known Security Considerations

### Model Safety

- **Prompt Injection**: The model may be susceptible to prompt injection attacks
- **Bias & Harmful Content**: Model outputs may contain biases or generate harmful content
- **Data Memorization**: Models may memorize and reproduce training data

### System Security

- **GPU Access**: Training requires GPU access which may expose system resources
- **File System Access**: Scripts have access to file system for checkpoints and data
- **Memory Usage**: Large models may exhaust system memory if not properly configured

## Security Updates

Security updates will be documented in:
- `docs/CHANGELOG.md` - All changes including security fixes
- GitHub Security Advisories - Critical vulnerabilities
- Release notes - Version-specific security improvements

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve Atlas security (with permission).

---

**Last Updated**: December 7, 2025
