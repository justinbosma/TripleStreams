# loaders and samplers
from model.Base.utils import get_hits_activation

# BasicGrooveTransformer imports
from model.Base.BasicGrooveTransformer import GrooveTransformer
from model.Base.BasicGrooveTransformer import GrooveTransformerEncoder


# VAE Imports
import model.VAE.shared_model_components as VAE_components
from model.VAE.MonotonicGrooveVAE import GrooveTransformerEncoderVAE

# CompGenVAE Imports
import model.CompGenVAE.components as CompGenComponents
from model.CompGenVAE.CompGenVAE import ComplexityGenreVAE

# GenVAE Imports
import model.GenVAE.components as GenreComponents
from model.GenVAE.GenVAE import GenreVAE

# GenDensityTempoVAE Imports
import model.GenDensityTempoVAE.components as GenDensityTempoComponents
from model.GenDensityTempoVAE.GenDensityTempoVAE import GenreDensityTempoVAE

# GenGlobalDenMuteVAE Imports
import model.GenGlobalDenMuteVAE.components as GenGlobalDenMuteComponents
from model.GenGlobalDenMuteVAE.GenGlobalDenMuteVAE import GenreGlobalDensityWithVoiceMutesVAE

# GenGlobalDenMuteVAE Imports
import model.GenMuteVAE.components as GenreWithVoiceMutesVAEComponents
from model.GenMuteVAE.GenMuteVAE import GenreWithVoiceMutesVAE

# GenMuteVAEMultiTask Imports
import model.GenMuteVAEMultiTask.components as GenreWithVoiceMutesMultiTaskComponents
from model.GenMuteVAEMultiTask.GenMuteVAEMultiTask import GenreWithVoiceMutesMultiTaskVAE

# VoiceMutesMultiTaskVAE Imports
import model.MuteVAEMultiTask.components as VoiceMutesMultiTaskVAEComponents
from model.MuteVAEMultiTask.VoiceMutesMultiTaskVAE import VoiceMutesMultiTaskVAE
