# Master Thesis Documentation

This directory contains the master thesis document and related materials.

## Thesis Information

**Title**: Utilizing Graph and Reinforcement Learning Algorithms for SSH Key Retrieval from Heap Memory Dumps

**Author**: Cyril GOMES  
**Institution**: Universität Passau  
**Supervisors**:
- Prof. Dr. Michael Granitzer
- Prof. Dr. Harald Kosch

## Research Motivation

SSH (Secure Shell) is a fundamental protocol for secure remote communication. However, its security can be exploited by malicious actors to create hidden, encrypted backdoors. A recent example is the backdoor discovered in the XZ/liblzma library targeting OpenSSH server processes. Traditional forensic methods can be intrusive or require prior knowledge of software versions. This work addresses these limitations by extracting temporary session keys from process memory dumps, enabling offline analysis without alerting potential attackers.

## Research Questions

1. Is it possible to extract SSH keys from a heap dump when very little training data is present?
2. Is there a suitable data structure that can represent the semantic relationships within the structures in a heap?

## Contents

- `master_thesis_cyril_gomes.pdf`: Full thesis document
- Additional materials and resources

## Implementation

The implementation is organized in two branches:
- `main`: Clean implementation of the thesis code
- `experiments`: Experimental iterations and research explorations

## Citation

```bibtex
@mastersthesis{gomes2025rlkex,
  author  = {Gomes, Cyril},
  title   = {Utilizing Graph and Reinforcement Learning Algorithms for SSH Key Retrieval from Heap Memory Dumps},
  school  = {Universität Passau},
  year    = {2025},
  address = {Passau, Germany},
  month   = {June}
}
```
