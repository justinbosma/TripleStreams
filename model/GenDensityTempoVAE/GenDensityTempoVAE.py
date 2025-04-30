#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import json
import os

from model import GenDensityTempoComponents
class GenreDensityTempoVAE(torch.nn.Module):
    """
    An encoder-encoder VAE transformer
    """
    def __init__(self, config):
        """
        This is a VAE transformer which for encoder and decoder uses the same transformer architecture
        (that is, uses the Vanilla Transformer Encoder)
        :param config: a dictionary containing the following keys:
            d_model_enc: the dimension of the model for the encoder
            d_model_dec: the dimension of the model for the decoder
            embedding_size_src: the dimension of the input embedding
            embedding_size_tgt: the dimension of the output embedding
            nhead_enc: the number of heads for the encoder
            nhead_dec: the number of heads for the decoder
            dim_feedforward_enc: the dimension of the feedforward network in the encoder
            dim_feedforward_dec: the dimension of the feedforward network in the decoder
            num_encoder_layers: the number of encoder layers
            num_decoder_layers: the number of decoder layers
            dropout: the dropout rate
            latent_dim: the dimension of the latent space
            max_len_enc: the maximum length of the input sequence
            max_len_dec: the maximum length of the output sequence
            device: the device to use
        """

        super(GenreDensityTempoVAE, self).__init__()

        assert config['embedding_size_tgt'] % 3 == 0, 'embedding_size_tgt must be divisible by 3'

        self.config = config
        self.latent_dim = config['latent_dim']
        self.n_genres = config['n_genres']

        # HParams
        # ---------------------------------------------------
        # self.d_model_enc = config['d_model_enc']
        # self.d_model_dec = config['d_model_dec']
        # self.embedding_size_src = config['embedding_size_src']
        # self.embedding_size_tgt = config['embedding_size_tgt']
        # self.nhead_enc = config['nhead_enc']
        # self.nhead_dec = config['nhead_dec']
        # self.dim_feedforward_enc = config['dim_feedforward_enc']
        # self.dim_feedforward_dec = config['dim_feedforward_dec']
        # self.num_encoder_layers = config['num_encoder_layers']
        # self.num_decoder_layers = config['num_decoder_layers']
        # self.dropout = config['dropout']
        # latent_dim = config['latent_dim']
        # self.max_len_enc = config['max_len_enc']
        # self.max_len_dec = config['max_len_dec']
        # self.velocity_dropout = config['velocity_dropout']
        # self.offset_dropout = config['offset_dropout']
        # self.n_genres = config['n_genres']
        # self.device = config['device']
        # self.n_density_bins = config['n_density_bins']
        # self.num_tempo_bins = config['num_tempo_bins']

        # Layers for the Groove2Drum VAE
        # ---------------------------------------------------
        self.InputLayerEncoder = GenDensityTempoComponents.InputGrooveLayer(
            embedding_size=self.config['embedding_size_src'],
            d_model=self.config['d_model_enc'],
            max_len=self.config['max_len_enc'],
            velocity_dropout=float(self.config['velocity_dropout']),
            offset_dropout=float(self.config['offset_dropout']),
            positional_encoding_dropout=float(self.config['dropout'])
        )

        self.Encoder = GenDensityTempoComponents.Encoder(
            d_model=self.config['d_model_enc'],
            nhead=self.config['nhead_enc'],
            dim_feedforward=self.config['dim_feedforward_enc'],
            num_encoder_layers=self.config['num_encoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.latentLayer = GenDensityTempoComponents.LatentLayer(
            max_len=self.config['max_len_enc'],
            d_model=self.config['d_model_enc'],
            latent_dim=self.config['latent_dim']
        )

        self.DecoderInput = GenDensityTempoComponents.DecoderInput(
            max_len=self.config['max_len_dec'],
            latent_dim=self.config['latent_dim'],
            d_model=self.config['d_model_dec'],
            n_genres=self.config['n_genres'],
            n_density_bins=self.config['n_density_bins'],
            n_tempo_bins=self.config['n_tempo_bins']
        )

        self.Decoder = GenDensityTempoComponents.Encoder(
            d_model=self.config['d_model_dec'],
            nhead=self.config['nhead_dec'],
            dim_feedforward=self.config['dim_feedforward_dec'],
            num_encoder_layers=self.config['num_decoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.OutputLayer = GenDensityTempoComponents.OutputLayer(
            embedding_size=self.config['embedding_size_tgt'],
            d_model=self.config['d_model_dec'],
        )

        # Initialize weights and biases
        self.InputLayerEncoder.init_weights()
        self.latentLayer.init_weights()
        self.DecoderInput.init_weights()
        self.OutputLayer.init_weights()


    @torch.jit.export
    def get_latent_dim(self):
        return self.latent_dim

    @torch.jit.export
    def encodeLatent(self, flat_hvo_groove: torch.Tensor):
        """
        Encodes the input sequence through the encoder and predicts the latent space

        :param flat_hvo_groove: [N, 32, 3]
        :return: mu, log_var, latent_z, complexity_logits, genre_logits
                mu:            [N, latent_dim]
                log_var:       [N, latent_dim]
                latent_z:      [N, latent_dim]
                memory:        [N, 32, d_model_enc]

        """
        x, hit, hvo_projection = self.InputLayerEncoder(hvo=flat_hvo_groove)
        memory = self.Encoder(x)  # N x (32+1) x d_model_enc
        mu, log_var, latent_z = self.latentLayer(memory)
        return mu, log_var, latent_z, memory

    @torch.jit.export
    def encode_all(self, flat_hvo_groove: torch.Tensor):
        """
        just here to make the model compatible with the previous CompGenVAE (in VST)
        """
        return self.encodeLatent(flat_hvo_groove)

    @torch.jit.export
    def decode(self, latent_z: torch.Tensor, genre_tags: torch.Tensor, tempo_bins: torch.Tensor,
               kick_density_bins: torch.Tensor, snare_density_bins: torch.Tensor,
               hats_density_bins: torch.Tensor, toms_density_bins: torch.Tensor,
               cymbals_density_bins: torch.Tensor):
        """
        Decodes the latent_z (N x latent_dim)  (through the decoder

        This one expects that genre are activated (softmaxed / sigmoided)

        :param latent_z:            [N, latent_dim]
        :param genre:               [N, 1] 
        :param tempo_bins:          [N, num_tempo_bins]
        :param kick_density_bins:   [N, n_density_bins]
        :param snare_density_bins:  [N, n_density_bins]
        :param hats_density_bins:   [N, n_density_bins]
        :param toms_density_bins:   [N, n_density_bins]
        :param cymbals_density_bins: [N, n_density_bins]
        :return:                    h_logits, v_logits, o_logits, hvo_logits

                h_logits: [N, 32, 1]
                v_logits: [N, 32, 1]
                o_logits: [N, 32, 1]
                hvo_logits: [N, 32, 27]

                None of the returned logits are activated (no sigmoid applied)
        """

        dec_in_ = self.DecoderInput.forward(
            latent_z=latent_z,
            genre_tags=genre_tags,
            tempo_bins=tempo_bins,
            kick_density_bins=kick_density_bins,
            snare_density_bins=snare_density_bins,
            hats_density_bins=hats_density_bins,
            toms_density_bins=toms_density_bins,
            cymbals_density_bins=cymbals_density_bins
        )

        dec_out_ = self.Decoder.forward(dec_in_)

        h_logits, v_logits, o_logits, hvo_logits = self.OutputLayer.forward(dec_out_)

        return h_logits, v_logits, o_logits, hvo_logits

    @torch.jit.export
    def sample(self,
               latent_z: torch.Tensor,
               genre: torch.Tensor,
               tempo_bins: torch.Tensor,
               kick_density_bins: torch.Tensor,
               snare_density_bins: torch.Tensor,
               hats_density_bins: torch.Tensor,
               toms_density_bins: torch.Tensor,
               cymbals_density_bins: torch.Tensor,
               voice_thresholds: torch.Tensor,
               voice_max_count_allowed: torch.Tensor,
               sampling_mode: int = 0):

        h_logits, v_logits, o_logits, hvo_logits = self.decode(
            latent_z=latent_z,
            genre_tags=genre,
            tempo_bins=tempo_bins,
            kick_density_bins=kick_density_bins,
            snare_density_bins=snare_density_bins,
            hats_density_bins=hats_density_bins,
            toms_density_bins=toms_density_bins,
            cymbals_density_bins=cymbals_density_bins
        )

        _h = torch.sigmoid(h_logits)
        v = torch.sigmoid(v_logits)
        o = torch.tanh(o_logits)

        h = torch.zeros_like(_h)

        if sampling_mode == 0:
            for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                max_indices = torch.topk(_h[:, :, ix], max_count).indices[0]
                h[:, max_indices, ix] = _h[:, max_indices, ix]
                h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)

        elif sampling_mode == 1:
            for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                # sample using probability distribution of hits (_h)
                voice_probs = _h[:, :, ix]
                sampled_indices = torch.bernoulli(voice_probs)
                max_indices = torch.topk(sampled_indices * voice_probs, max_count).indices[0]
                h[:, max_indices, ix] = 1

        return h, v, o

    @torch.jit.export
    def predict(self,
                flat_hvo_groove: torch.Tensor,
                genre_tags: torch.Tensor,
                tempo_bins: torch.Tensor,
                kick_density_bins: torch.Tensor,
                snare_density_bins: torch.Tensor,
                hats_density_bins: torch.Tensor,
                toms_density_bins: torch.Tensor,
                cymbals_density_bins: torch.Tensor,
                threshold: float=0.5):

        h_logits, v_logits, o_logits, mu, log_var, latent_z = self.forward(
            flat_hvo_groove=flat_hvo_groove,
            genre_tags=genre_tags,
            tempo_bins=tempo_bins,
            kick_density_bins=kick_density_bins,
            snare_density_bins=snare_density_bins,
            hats_density_bins=hats_density_bins,
            toms_density_bins=toms_density_bins,
            cymbals_density_bins=cymbals_density_bins
        )

        _h = torch.sigmoid(h_logits)
        v = torch.sigmoid(v_logits)
        o = torch.tanh(o_logits)

        h = torch.zeros_like(_h)

        voice_thresholds = torch.tensor([threshold] * 9)
        voice_max_count_allowed = torch.tensor([32] * 9)

        for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
            max_indices = torch.topk(_h[:, :, ix], max_count).indices[0]
            h[:, max_indices, ix] = _h[:, max_indices, ix]
            h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)

        hvo = torch.cat([h, v, o], dim=-1)

        return hvo, latent_z

    @torch.jit.export
    def forward(self, flat_hvo_groove: torch.Tensor, genre_tags: torch.Tensor, tempo_bins: torch.Tensor,
                kick_density_bins: torch.Tensor, snare_density_bins: torch.Tensor, hats_density_bins: torch.Tensor,
                toms_density_bins: torch.Tensor, cymbals_density_bins: torch.Tensor):

        mu, log_var, latent_z, memory = self.encodeLatent(flat_hvo_groove)

        h_logits, v_logits, o_logits, hvo_logits = self.decode(
            latent_z=latent_z, genre_tags=genre_tags, tempo_bins=tempo_bins,
            kick_density_bins=kick_density_bins, snare_density_bins=snare_density_bins,
            hats_density_bins=hats_density_bins, toms_density_bins=toms_density_bins,
            cymbals_density_bins=cymbals_density_bins)

        return h_logits, v_logits, o_logits, mu, log_var, latent_z

    @torch.jit.ignore
    def save(self, save_path, additional_info=None):
        """ Saves the model to the given path. The Saved pickle has all the parameters ('params' field) as well as
        the state_dict ('state_dict' field) """
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        config_ = dict()
        for key, value in self.config.items():
            config_[key] = value
        json.dump(config_, open(save_path.replace('.pth', '.json'), 'w'))
        torch.save({'model_state_dict': self.state_dict(), 'params': config_,
                    'additional_info': additional_info}, save_path)

    # serializes to a torchscript model
    @torch.jit.ignore
    def serialize(self, save_folder, filename=None):

        os.makedirs(save_folder, exist_ok=True)

        if filename is None:
            import datetime
            filename = f'GenDensTempVAE_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        is_train = self.training
        self.eval()
        save_path = os.path.join(save_folder, filename)

        scr = torch.jit.script(self)
        # save model
        with open(save_path, "wb") as f:
            torch.jit.save(scr, f)

        if is_train:
            self.train()


if __name__ == "__main__":

    config = {
        'd_model_enc': 128,
        'd_model_dec': 128,
        'embedding_size_src': 3,
        'embedding_size_tgt': 27,
        'nhead_enc': 4,
        'nhead_dec': 8,
        'dim_feedforward_enc': 128,
        'dim_feedforward_dec': 512,
        'num_encoder_layers': 3,
        'num_decoder_layers': 6,
        'dropout': 0.1,
        'latent_dim': 16,
        'max_len_enc': 32,
        'max_len_dec': 32,
        'velocity_dropout': 0.1,
        'offset_dropout': 0.2,
        'n_genres': 12,
        'n_density_bins': 3,
        'n_tempo_bins': 7,
        'device': 'cpu'
    }

    model = GenreDensityTempoVAE(config)
    kick_density_bins = torch.tensor([1])
    snare_density_bins = torch.tensor([[1]])
    hats_density_bins = torch.tensor([[1]])
    toms_density_bins = torch.tensor([[1]])
    cymbals_density_bins = torch.tensor([[1]])
    tempo_bins = torch.tensor([[3]])
    genre_tags = torch.tensor([[1]])
    model.forward(torch.rand(1, 32, 3), genre_tags, tempo_bins, kick_density_bins, snare_density_bins, hats_density_bins,
                  toms_density_bins, cymbals_density_bins)
    model.decode(torch.rand(1, 16), genre_tags, tempo_bins, kick_density_bins, snare_density_bins, hats_density_bins,
                 toms_density_bins, cymbals_density_bins)
    model.serialize(save_folder='./')
