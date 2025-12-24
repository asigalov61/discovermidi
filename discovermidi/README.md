# Discover MIDI Dataset
## Ultimate MIDI dataset for MIDI music discovery and symbolic music AI purposes

<img width="1024" height="1024" alt="Discover-MIDI-Dataset" src="https://github.com/user-attachments/assets/a729b21a-e666-4996-8f61-78ae8c589ba1" />

***

### Introduction

#### Bright, comprehensive, and built for discovery — the **Discover MIDI Dataset** is a massive, carefully curated collection of symbolic music designed for music information retrieval (MIR), creative exploration, and training symbolic music AI. It contains **over 6.74M unique, de‑duplicated, and normalized MIDI files**, comprehensive metadata, and GPU‑accelerated search tooling so researchers and creators can find, analyze, and prototype with MIDI at production scale.

### Abstract

_The **Discover MIDI Dataset** is a large‑scale, production‑ready collection of symbolic music designed for music information retrieval, discovery, and symbolic music AI. It aggregates **over 6.74 million** unique, de‑duplicated, and normalized MIDI files, each validated against the MIDI specification and integrity‑checked to ensure consistent, high‑quality inputs for analysis and model development. The dataset emphasizes reproducibility and efficiency by applying a two‑stage de‑duplication pipeline (MD5 hashing and pitch/chord count comparison) and by providing extensive, precomputed metadata and feature representations for every file._

_The dataset supplies **rich, structured metadata** including feature counts, compressed features matrixes (961 features excluding velocities), file lists, genre labels, artist/title identifications, karaoke and lyrics matches, monophonic melody summaries, pitches‑patches counts, and detailed quality metrics for alignment, chords, durations, and pitches. Features span a unified index range covering delta start times, durations, instruments/patches, instrument and drum pitches, a 321‑chord harmonic vocabulary, and velocities, enabling consistent statistical analysis, de‑duplication, and retrieval tasks across millions of files._

_To accelerate discovery workflows, Discover MIDI includes a **GPU‑accelerated search and filter engine** optimized for float16 performance; on suitable hardware (recommended ≥16 GB GPU VRAM) full‑dataset similarity searches complete in the order of **10–20 seconds per master MIDI**. The distribution also bundles supplemental code—such as a MIDI loops extractor—and curated soundfont banks to support rendering, loop extraction, and downstream experimentation. Convenience artifacts like MD5→path file lists and curated subsets make large‑scale batch processing and targeted retrieval straightforward._

_The dataset is packaged with clear installation and optional dependency instructions for CPU and GPU environments, plus modules to speed parallel extraction and audio rendering. Its combination of scale, rigorous de‑duplication, precomputed features, and high‑performance tooling makes Discover MIDI suitable for a wide range of use cases: training and evaluating symbolic music models, large‑scale MIR research, motif and loop discovery for creators, and building retrieval or recommendation systems with minimal preprocessing overhead._

### Overview

- **Purpose:** Large‑scale symbolic dataset for MIR, retrieval, analysis, and generative model development.  
- **Scale:** 6.74M+ unique MIDIs, each converted to a proper MIDI specification and integrity‑checked.  
- **Quality controls:** Two‑stage de‑duplication (MD5 and pitch/chord counts) and extensive quality metrics.  
- **Tooling:** Precomputed features, compressed features matrixes, and a custom GPU‑accelerated search and filter codebase.

### Key features

- **Massive, de‑duplicated collection** — over **6.74M** normalized MIDIs with integrity checks.  
- **Two‑stage de‑duping** — MD5 hash deduplication followed by pitch/chord counts deduplication.  
- **Rich metadata** — features counts, features matrixes, file lists, genre mappings, identified artist/title, karaoke and lyrics matches, mono‑melody info, pitches‑patches counts, and quality metrics.  
- **Precomputed features** — features span indices \([0,1089)\) grouped into delta start times, durations, instruments/patches, pitches, harmonic chords (321 chords), and velocities; matrixes exclude velocities and are stored as compressed NumPy arrays (961 features).  
- **High‑performance search** — GPU‑accelerated search and filter code optimized for float16; full searches typically take **10–20 seconds per master MIDI** on a capable GPU (recommended **≥16 GB VRAM**).  
- **Convenience files** — curated file lists (md5 → full path) and many subsets for easy retrieval and batch processing.  
- **Supplemental code** — MIDI loops extractor, rendering helpers, and optional modules to speed extraction and audio rendering.

***

## Installation

### pip and setuptools

```sh
# It is recommended that you upgrade pip, setuptools, build and wheel prior to install for max compatibility
!pip install --upgrade pip
!pip install --upgrade setuptools build wheel
```

### CPU/GPU install

#### Auto-install with pip

```sh
# The following command will install Discover MIDI Dataset for fast GPU search only
# Please note that GPU search requires at least 16GB GPU VRAM for fast full searches
!pip install -U discovermidi

# Alternativelly, you can use [full] option to install dependencies for all included modules
!pip install -U discovermidi[full]
```

#### Manual dependencies installation

```sh
# Core dependencies for search and filter modules
!pip install tqdm
!pip install ipywidgets
!pip install scikit-learn
!pip install scipy
!pip install matplotlib
!pip install hf-transfer
!pip install huggingface_hub
!pip install torch
!pip install midirenderer
!pip install mididoctor
!pip install numpy==1.24.4
```

```sh
# Dependencies for midi_loops_extractor modules
!pip install pretty-midi
!pip install symusic
!pip install miditok
!pip install numba
```

```sh
# Dependencies for aux modules
!pip install networkx
!pip install cupy-cuda13x
!pip install einops
!pip install einx
!pip install nltk
```

### Optional packages

#### Packages for fast_parallel_extract module

```sh
# The following command will install packages for fast_parallel_extract module
# It will allow you to extract (untar) Discover MIDI Dataset much faster
!sudo apt update -y
!sudo apt install -y p7zip-full
!sudo apt install -y pigz
```

#### Packages for midi_to_colab_audio module

```sh
# The following command will install packages for midi_to_colab_audio module
# It will allow you to render Discover MIDI Dataset MIDIs to audio
!sudo apt update -y
!sudo apt install fluidsynth
```

#### Packages for midi_loops_extractor codebase

```sh
# The following command will install additional packages for midi_loops_extractor codebase
# It will allow you to extract loops from Discover MIDI Dataset MIDIs
# Please see README.md in discovermidi/midi_loops_extractor/ for use instructions
!pip install -U discovermidi[loops]
```

```sh
# Alternativelly, you can install dependencise for loops codebase manually
!pip install pretty-midi
!pip install symusic
!pip install miditok
!pip install numba
!pip install numpy==1.24.4
```

***

## Quick-start use example

```python
# Import main Discover MIDI Dataset module
import discovermidi

# Download Discover MIDI Dataset from Hugging Face repo
discovermidi.download_dataset()

# Extract Discover MIDI Dataset with built-in function (slow)
discovermidi.parallel_extract()

# Or you can extract much faster if you have installed the optional packages for Fast Parallel Extract module
# from discovermidi import fast_parallel_extract
# fast_parallel_extract.fast_parallel_extract()

# Load all MIDIs features matrixes and their corresponding MIDIs file names
features_matrixes, features_matrixes_file_names = discovermidi.load_features_matrixes()

# Run the search
# IO dirs will be created on the first run of the following function
# Do not forget to put your master MIDIs into created Master-MIDI-Dataset folder
# The full search for each master MIDI takes about 10-25 seconds on a GPU
discovermidi.search_and_filter(features_matrixes, features_matrixes_file_names)
```

***

## Dataset structure information

```
Discover-MIDI-Dataset/              # Dataset root dir
├── ARTWORK/                        # Concept artwork
│   ├── Illustrations/              # Concept illustrations
│   ├── Logos/                      # Dataset logos
│   └── Posters/                    # Dataset posters
├── CODE/                           # Root dir for supplemental python code and python modules
│   └── midi_loops_extractor/       # MIDI loops extractor codebase dir
├── DATA/                           # Dataset (meta)data dir
│   ├── Features Counts/            # Features counts for all MIDIs
│   ├── Features Matrixes/          # Pre-computed compressed Features counts matrixes for all MIDIs
│   ├── Files Lists/                # Files lists by MIDIs types and categories
│   ├── Genres MIDIs/               # Genres, artists and titles data for all matched MIDIs
│   ├── Identified MIDIs/           # Comprehensive data for identified MIDIs
│   ├── Karaoke MIDIs/              # Karaoke MIDIs data
│   ├── Lyrics MIDIs/               # Lyrics for matched MIDIs
│   ├── Mono Melodies/              # Data for all MIDIs with monophonic melodies
│   ├── Pitches Patches Counts/     # Pitches-patches counts for all MIDIs
│   └── Quality/                    # Quality data for most MIDIs
├── MIDIs/                          # Root MIDI files dir
└── SOUNDFONTS/                     # Select high-quality Sound Fonts banks to render MIDIs
```

***

## Dataset (meta)data information

****

### Features Counts

#### Features counts for all MIDIs are presented in a form of list of tuples (feature, count)

#### Features range is [0-1089) which covers six groups of values

* ##### [0-128) Delta start times
* ##### (128-256) Durations
* ##### [256-384] MIDI patches/instruments, 384 being reserved for drums
* ##### (384-640) MIDI pitches: (384-512) reserved for instruments and (512-640) for drums
* ##### [640-961) All possible harmonic chords (321 chords)
* ##### (961-1089) Velocities

****

### Features Matrixes

#### A compressed NumPy array of flattened features matrixes, covering 961 out of 1089 features (without velocities features)

****

### Files lists

#### Numerous files lists were created for convenience and easy MIDIs retrieval from the dataset
#### These include lists of all MIDIs as well as subsets of MIDIs
#### Files lists are presented in a dictionary format of two strings:

* ##### MIDI md5 hash
* ##### Full MIDI path

****

### Genres MIDIs

#### This data contains information about all MIDIs that were definitively identified by music genre

****

### Identified MIDIs

#### This data contains information about all MIDIs that were definitively identified by artist and title

****

### Karaoke MIDIs

#### This data contains information about all MIDIs that were definitively identified as Karaoke

****

### Lyrics MIDIs

#### This data contains information about all MIDIs that were definitively matched to corresponding lyrics

****

### Mono melodies

#### This data contains information about all MIDIs with at least one monophonic melody
#### The data in a form of list of tuples where first element represents monophonic melody patch/instrument
#### And the second element of the tuple represents number of notes for indicated patch/instrument
#### Please note that many MIDIs may have more than one monophonic melody

****

### Pitches patches counts

#### This data contains the pitches-patches counts for all MIDIs in the dataset
#### This information is very useful for de-duping, MIR and statistical analysis

****

### Quality

#### This data contains detailed quality information for most MIDIs in the dataset
#### Each data entry contains information about quality of MIDI alignment, chords, durations, pitches and type

****

## Citations

```bibtex
@misc{project_los_angeles_2025,
	author       = { Project Los Angeles },
	title        = { Discover-MIDI-Dataset },
	year         = 2025,
	url          = { https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset },
	publisher    = { Hugging Face }
}
```

```bibtex
@misc{DiscoverMIDIDataset2025,
  title        = {Godzilla MIDI Dataset: Enormous, comprehensive, normalized and searchable MIDI dataset for MIR and symbolic music AI purposes},
  author       = {Alex Lev},
  publisher    = {Project Los Angeles / Tegridy Code},
  year         = {2025},
  url          = {https://huggingface.co/datasets/projectlosangeles/Godzilla-MIDI-Dataset}
```

```bibtex
@misc {breadai_2025,
    author       = { {BreadAi} },
    title        = { Sourdough-midi-dataset (Revision cd19431) },
    year         = 2025,
    url          = {\url{https://huggingface.co/datasets/BreadAi/Sourdough-midi-dataset}},
    doi          = { 10.57967/hf/4743 },
    publisher    = { Hugging Face }
}
```

```bibtex
@inproceedings{bradshawaria,
  title={Aria-MIDI: A Dataset of Piano MIDI Files for Symbolic Music Modeling},
  author={Bradshaw, Louis and Colton, Simon},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=X5hrhgndxW}, 
}
```

```bibtex
@misc{TegridyMIDIDataset2025,
  title        = {Tegridy MIDI Dataset: Ultimate Multi-Instrumental MIDI Dataset for MIR and Music AI purposes},
  author       = {Alex Lev},
  publisher    = {Project Los Angeles / Tegridy Code},
  year         = {2025},
  url          = {https://github.com/asigalov61/Tegridy-MIDI-Dataset}
```

***

### Project Los Angeles
### Tegridy Code 2025
