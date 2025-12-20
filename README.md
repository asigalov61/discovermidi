# Discover MIDI Dataset
## Ultimate MIDI dataset for MIDI music discovery and symbolic music AI purposes

<img width="1024" height="1024" alt="Discover-MIDI-Dataset" src="https://github.com/user-attachments/assets/a729b21a-e666-4996-8f61-78ae8c589ba1" />

***
***

## Dataset features

### 1) Over 6.74M+ unique, de-duped and normalized MIDIs
### 2) Each MIDI was converted to proper MIDI format specification and checked for integrity
### 3) Dataset was de-duped twice: by md5 hashes and by the pitches-chords counts
### 4) Extensive and comprehensive (meta)data was collected from all MIDIs in the dataset
### 5) Dataset comes with a custom-designed and highly optimized GPU-accelerated search and filter code

***

## Installation

### pip and setuptools

```sh
# It is recommended that you upgrade pip and setuptools prior to install for max compatibility
!pip install --upgrade pip
!pip install --upgrade setuptools
```

### CPU/GPU install

```sh
# The following command will install Discover MIDI Dataset for fast GPU search
# Please note that GPU search requires at least 16GB GPU VRAM for full searches at float16 precision

!pip install -U discovermidi
```

### Optional packages

#### Packages for Fast Parallel Extract module

```sh
# The following command will install packages for Fast Parallel Extract module
# It will allow you to extract (untar) Godzilla MIDI Dataset much faster

!sudo apt update -y
!sudo apt install -y p7zip-full
!sudo apt install -y pigz
```

#### Packages for midi_to_colab_audio module

```sh
# The following command will install packages for midi_to_colab_audio module
# It will allow you to render Godzilla MIDI Dataset MIDIs to audio

!sudo apt update -y
!sudo apt install fluidsynth
```

***

## Quick-start use example

```python
# Import main Godzilla MIDI Dataset module
import discovermidi

# Download Godzilla MIDI Dataset from Hugging Face repo
discovermidi.download_dataset()

# Extract Godzilla MIDI Dataset with built-in function (slow)
discovermidi.parallel_extract()

# Or you can extract much faster if you have installed the optional packages for Fast Parallel Extract
# from discovermidi import fast_parallel_extract
# fast_parallel_extract.fast_parallel_extract()

# Load all MIDIs features matrixes and their corresponding MIDIs file names
features_matrixes, features_matrixes_file_names = discovermidi.load_features_matrixes()

# Run the search
# IO dirs will be created on the first run of the following function
# Do not forget to put your master MIDIs into created Master-MIDI-Dataset folder
# The full search for each master MIDI takes about 10-20 seconds on a GPU
discovermidi.search_and_filter(features_matrixes, features_matrixes_file_names)
```

***

## Dataset structure information

```
Discover-MIDI-Dataset/               # Dataset root dir
├── ARTWORK/                        # Concept artwork
│   ├── Illustrations/              # Concept illustrations
│   ├── Logos/                      # Dataset logos
│   └── Posters/                    # Dataset posters
├── CODE/                           # Root dir for supplemental python code and python modules
│   ├── midi_loops_extractor/       # MIDI loops extractor code dir
├── DATA/                           # Dataset (meta)data dir
│   ├── Features Counts/            # Features counts for each MIDI
│   ├── Features Matrixes/          # Features counts matrixes for each MIDI
│   ├── Files Lists/                # Files lists by MIDIs types and categories
│   ├── Identified MIDIs/           # Comprehensive data for identified MIDIs
│   ├── Karaoke MIDIs/              # Karaoke MIDIs data
│   ├── MIDIs Lyrics/               # MIDIs lyrics data
│   ├── Mono Melodies/              # Data for all MIDIs with monophonic melodies
│   ├── Pitches Patches Counts/     # Pitches-patches counts for all MIDIs 
├── MIDIs/                          # Root MIDIs dir
└── SOUNDFONTS/                     # Select high-quality Sound Font banks to render MIDIs
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

#### A compressed NumPy array of flattened features matrixes, covering 961 out of 1089 features (without velocities)

****

### Files lists

#### Numerous files lists were created for convenience and easy MIDIs retrieval from the dataset
#### These include lists of all MIDIs as well as subsets of MIDIs
#### Files lists are presented in a dictionary format of two strings:

* ##### MIDI md5 hash
* ##### Full MIDI path

****

### Identified MIDIs

#### This data contains information about all MIDIs that were definitively identified by artist and title

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

## Citations

```bibtex
@misc{GodzillaMIDIDataset2025,
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