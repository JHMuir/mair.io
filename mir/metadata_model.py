from typing_extensions import Annotated, List, Optional
from pydantic import BaseModel, RootModel, Field


def get_schema_descriptions() -> dict:
    schema = AudioMetadata.model_json_schema()
    descriptions = {}
    for name, field in schema.get("properties", {}).items():
        if "description" in field:
            descriptions[name] = field["description"]
    return descriptions


class AudioMetadata(BaseModel):
    """Pydantic model for audio metadata with field descriptors"""

    # Required Values
    sampling_rate: Annotated[
        int, Field(description="Audio sampling rate in Hz", ge=8000, le=48000)
    ]
    tempo: Annotated[
        int, Field(description="Tempo in beats per minute (BPM)", ge=50, le=300)
    ]
    rhythm_regularity: Annotated[
        float,
        Field(
            description="Measure of rhythm consistency (std/mean of tempo structure). Lower values indicate more regular patterns",
            ge=1.0,
            le=15.0,
        ),
    ]
    spectral_centroid_mean: Annotated[
        float,
        Field(
            description="Average frequency center of the spectrum. Higher values indicate brighter sounds",
            ge=50,
            le=3000,
        ),
    ]
    spectral_contrast_mean: Annotated[
        List[float],
        Field(description="Mean spectral contrast values across frequency bands"),
    ]
    bass_contrast: Annotated[
        float,
        Field(
            description="Contrast in the bass frequencies. Higher values indicate stronger bass presence",
            ge=10,
            le=30,
        ),
    ]
    treble_contrast: Annotated[
        float,
        Field(
            description="Contrast in the treble frequencies. Higher values indicate brighter treble",
            ge=5,
            le=40,
        ),
    ]
    energy_mean: Annotated[
        float,
        Field(
            description="Average energy/RMS of the audio. Higher values indicate louder or more consistent volume",
            ge=0.0,
            le=1.0,
        ),
    ]
    energy_std: Annotated[
        float,
        Field(
            description="Standard deviation of energy/RMS. Higher values indicate more dynamic range",
            ge=0.0,
            le=1.0,
        ),
    ]
    zero_crossing_rate_mean: Annotated[
        float,
        Field(
            description="Average rate of sign changes in the audio. Higher values indicate more high-frequency content or noise",
            ge=0.0,
            le=0.5,
        ),
    ]
    mfcc_profile: Annotated[
        List[float],
        Field(
            description="Average values for each MFCC coefficient, representing the timbre profile"
        ),
    ]
    low_mfcc: Annotated[
        float,
        Field(
            description="Average of lower MFCCs (1-4), related to timbre and bass characteristics",
            le=0,
        ),
    ]
    mid_mfcc: Annotated[
        float,
        Field(
            description="Average of middle MFCCs (5-9), related to timbral characteristics"
        ),
    ]
    high_mfcc: Annotated[
        float,
        Field(
            description="Average of higher MFCCs (10-13), related to high-frequency characteristics"
        ),
    ]
    mfcc_spread: Annotated[
        float,
        Field(
            description="Standard deviation of MFCC values, indicating timbral complexity",
            ge=0,
        ),
    ]
    tonal_features: Annotated[
        List[float],
        Field(description="Average tonnetz features representing harmonic content"),
    ]
    key: Annotated[
        str,
        Field(
            description="Estimated musical key of the audio based on Krumhansl-Schmuckler key profiles"
        ),
    ]
    complexity_score: Annotated[
        float,
        Field(
            description="Overall complexity measure based on spectral contrast. Higher values indicate more complex audio",
            ge=10,
            le=40,
        ),
    ]
    tonal_stability: Annotated[
        float,
        Field(
            description="Standard deviation of tonal features. Lower values indicate more stable tonality",
            ge=0.0,
            le=0.5,
        ),
    ]
    mood: Annotated[
        str,
        Field(
            description="Classified mood of the track based on energy, tempo, complexity, and tonality"
        ),
    ]
    function: Annotated[
        str,
        Field(
            "Categorized in-game function of the audio track. Multiple functions may be combined with commas"
        ),
    ]
    description: Annotated[
        str, Field("Human-readable description of the audio track's characteristics")
    ]

    # Optional Values (not included in the metadata.json but part of the extraction process)
    beat_times: Annotated[
        Optional[List[float]],
        Field(None, description="List of timestamps (in seconds) where beats occur"),
    ]
    beat_strength: Annotated[
        Optional[float],
        Field(
            None,
            description="Number of beats relative to track tempo, indicating rhythmic density",
        ),
    ]
    tempo_scores: Annotated[
        Optional[List[float]],
        Field(
            None,
            description="Scores for different candidate tempos, useful for identifying rhythm patterns",
        ),
    ]
    tempo_structure: Annotated[
        Optional[List[float]],
        Field(None, description="Temporal evolution of tempo patterns in the track"),
    ]
    chroma_mean: Annotated[
        Optional[List[float]],
        Field(
            None,
            description="Mean chromagram values for each pitch class, useful for key detection",
        ),
    ]


class AudioMetadataCollection(RootModel):
    root: Annotated[
        dict[str, AudioMetadata],
        Field(description="Dictionary mapping track filenames to their metadata"),
    ]

    def __getitem__(self, key: str) -> AudioMetadata:
        return self.root[key]

    def items(self):
        return self.root.items()

    def keys(self):
        """Get all track names"""
        return self.root.keys()

    def values(self):
        """Get all track metadata"""
        return self.__root__.values()
