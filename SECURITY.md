# Security Policy

## Project Overview

This repository contains the prototype implementation for **"Resource Analysis and Optimization for Big Data Analytics in Cloud Environments"** — a BScEng research project (EC6070) at the University of Jaffna, Department of Computer Engineering.

**Students:** Dayarathna D.D.R.N. (2022E033) | Lawanya M.A.S. (2022E090)  
**Supervisor:** Dr. J. Jananie

***

## Supported Versions

This is an **academic research prototype**. Only the latest version on the `main` branch is actively maintained and supported.

| Version | Supported |
|---------|-----------|
| Latest (`main` branch) | ✅ Yes |
| Older branches / tags | ❌ No |

***

## Scope

This security policy applies to the prototype codebase covering:

- Multi-driver Apache Spark execution architecture
- MPI coordination layer (mpi4py / MPJ-Express)
- Docker containerized cluster deployment
- NFS shared volume configuration
- ML workload profiling and resource allocation scripts

***

## Reporting a Vulnerability

If you discover a security vulnerability in this repository, please follow responsible disclosure practices:

1. **Do NOT open a public GitHub issue** for security vulnerabilities.
2. **Contact the maintainers directly** via the University of Jaffna Department of Computer Engineering, or email the project team privately.
3. Provide a clear description of:
   - The vulnerability and affected component
   - Steps to reproduce
   - Potential impact
   - Suggested mitigation (if known)

We will acknowledge your report within **72 hours** and aim to address confirmed vulnerabilities within **14 days**.

***

## Known Security Considerations

### Docker & Container Security

- Containers in this prototype run with **default Docker network settings**. For production use, network isolation (custom bridge networks or Kubernetes NetworkPolicy) should be enforced.
- NFS shared volumes are mounted without authentication in the development setup — **do not expose NFS mounts on public networks**.
- Docker images used are based on standard Apache Spark and Python base images. Regularly update base images to receive upstream security patches.

### MPI Communication (mpi4py / OpenMPI)

- MPI communication between containers uses **plain TCP** without encryption by default. For sensitive data, consider enabling OpenMPI's built-in SSL/TLS transport or routing through an encrypted overlay network.
- MPI rank authentication is not enforced in the prototype — restrict access to the Docker Swarm overlay network to trusted nodes only.

### Spark Configuration

- Spark Web UI (port `4040`) is exposed by default within the cluster. Do not expose this port publicly without authentication.
- Spark drivers communicate over configurable ports; ensure firewall rules restrict these to the internal Docker network.
- No Spark authentication (`spark.authenticate`) or encryption (`spark.network.encrypt`) is enabled in the prototype. These must be enabled before any deployment outside a trusted local network.

### Shared Storage (NFS)

- NFS exports in the prototype use permissive settings for ease of development. Before deploying in any shared or public environment, restrict NFS exports with proper `hosts allow`, `root_squash`, and read/write permission controls.

### Credentials and Secrets

- **No API keys, passwords, or cloud credentials** should be committed to this repository.
- Use `.gitignore` to exclude any environment files (`.env`, `config.yaml` containing secrets).
- If cloud credentials (AWS, GCP, Azure) are used for VM provisioning, use environment variables or secret management tools (e.g., Docker secrets, Kubernetes Secrets) — never hardcode them.

***

## Dependency Management

| Component | Recommendation |
|-----------|---------------|
| Apache Spark | Pin to a specific version in Dockerfile; update regularly |
| mpi4py | Install via `pip` with a pinned version in `requirements.txt` |
| Python base image | Use official `python:3.x-slim` and rebuild periodically |
| OpenMPI | Install from official package manager; avoid building from untrusted source |

Run dependency audits periodically:

```bash
pip audit
# or
pip list --outdated
```

***

## Responsible Use

This repository is intended **strictly for academic research and educational purposes**. The prototype is not designed for production workloads or handling sensitive personal data. Users who adapt this code for production environments are responsible for conducting their own security review and hardening.

***

## References

- [Apache Spark Security Documentation](https://spark.apache.org/docs/latest/security.html)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [OpenMPI Security Considerations](https://www.open-mpi.org/faq/?category=security)
- [mpi4py Documentation](https://mpi4py.readthedocs.io/)

***

*Last updated: June 2026*
