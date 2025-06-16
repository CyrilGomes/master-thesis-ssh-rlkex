# Master Thesis: Utilizing Graph and Reinforcement Learning for SSH Key Retrieval

This repository contains the implementation of my master thesis research at Universität Passau, focusing on extracting SSH session keys from heap memory dumps using Deep Reinforcement Learning and graph-based approaches.

## Author & Supervision
- **Author**: Cyril GOMES
- **Institution**: Universität Passau
- **Supervisors**: 
  - Prof. Dr. Michael Granitzer
  - Prof. Dr. Harald Kosch

## Code Organization
The repository is organized into branches:
- `main` (current): Clean implementation of the thesis code as used in the final experiments
- `experiments`: Contains experimental implementations and research iterations

## Abstract

This project introduces a novel approach for extracting SSH session keys directly from heap memory dumps of an OpenSSH process. The core of this method is the use of Deep Reinforcement Learning (DRL) to navigate a graph-based representation of the heap memory. This approach is designed to be highly effective even with limited training data and to generalize well to new, unseen data (such as new OpenSSH versions) with minimal or no retraining. The goal is to create a powerful digital forensics tool that can non-intrusively decrypt SSH communications to analyze malicious activity or monitor honeypots.

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.fim.uni-passau.de/gomes/ssh-rlkex.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://git.fim.uni-passau.de/gomes/ssh-rlkex/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

## Core Contributions

1. **Pointer Graph Generation Pipeline**: High-performance Rust implementation for converting heap dumps into pointer graphs
2. **Root Node Predictor**: Classifier for identifying optimal starting nodes (100% accuracy)
3. **Key Count Classifier**: GNN model for predicting SSH key counts (100% accuracy)
4. **Novel DRL Agent**: Goal-oriented Deep Q-Learning agent with 95% accuracy in key retrieval

## Methodology & Pipeline

1. **Dataset Processing**
   - Uses dataset from Fellicious et al. [1]
   - Raw heap dumps from various OpenSSH versions
   - JSON metadata with labeled key addresses

2. **Graph Generation**
   - Raw heap dump processing
   - Pointer identification and validation
   - Directed graph construction with memory block nodes
   - Node feature enrichment (struct_size, pointer counts, etc.)

3. **Graph Preprocessing & Helper Models**
   - GATv2Conv-based key count classifier
   - Random Forest root node predictor using SCC analysis

4. **Deep Reinforcement Learning**
   - Goal-oriented DDQN with Prioritized Experience Replay
   - Variable action space based on node connections
   - GATv2Conv layers for graph processing
   - BFS-based subgraph environment

## Project Structure

```
.
├── Graph_Gen/               # Memory dump to graph converter (Rust implementation)
│   └── graph_gen_rust/     # High-performance Rust implementation
├── Graph_Insights/         # Graph analysis and visualization tools
├── Heuristics/            # ML-based heuristics for key detection
├── RL/                    # Main RL implementation
│   └── GNNDRL/           # Graph Neural Network + Deep RL implementation
└── models/               # Pre-trained model checkpoints
```

## Installation

```bash
# Clone the repository
git clone https://github.com/CyrilGomes/master-thesis-ssh-rlkex.git
cd master-thesis-ssh-rlkex

# Create and activate conda environment
conda env create -f environment.yml
conda activate ssh-rlkex

# Install Rust for graph generation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Usage

### 1. Graph Generation
```bash
# Generate graph from heap dump
cd Graph_Gen/graph_gen_rust
cargo run --release -- /path/to/heap.dump /path/to/output/graph.graphml
```

### 2. Key Count Prediction
```bash
cd Heuristics
python nb_keys_classifier.py --data-path /path/to/graph.graphml
```

### 3. Root Node Prediction
```bash
cd Heuristics
python root_heuristic.py --input /path/to/graph.graphml
```

### 4. Key Retrieval
```bash
cd RL/GNNDRL/variable_action_space_goal
python graph_obs_variable_action_space_GDQL.py --eval \
    --model-path models/model.pt \
    --input-graph /path/to/graph.graphml
```

## Citation

If you use this code or find our work helpful, please cite:

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full license text.

## Key Results

1. **Root Predictor Performance**: 
   - 100% accuracy on test dataset
   - Successful classification of valid root nodes

2. **Key Count Prediction**: 
   - 100% accuracy on test dataset
   - Reliable prediction of 2, 4, or 6 keys

3. **DRL Agent Performance**:
   - 95% overall accuracy on validation dataset
   - High success rate across OpenSSH versions
   - Effective with limited training data (7 files per sub-folder)

## Technology Stack

- **Languages**: Python, Rust (graph generation)
- **Deep Learning**: PyTorch, PyTorch Geometric
- **Machine Learning**: Scikit-learn
- **Graph Processing**: NetworkX

## Publications and Links

- [Master Thesis PDF](docs/master_thesis_cyril_gomes.pdf)
- [Technical Documentation](docs/)

## References

[1] Fellicious et al., "Memory Dumps Analysis for Forensic Investigation", 2022.
