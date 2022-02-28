from module import *
from vae import VAE
from text.symbols import symbols
import hparams as hp
import random
import torchsnooper

class Encoder(nn.Module):
    def __init__(self, hidden_size):
        # 第一维是信号所有取值的数量，第二维是想embedding的目标维数
        super(Encoder, self).__init__()
        # print(embedding_size)
        self.embed = nn.Embedding(len(symbols), hp.embedding_size)
        self.prenet = Prenet(
            hp.embedding_size, hp.hidden_size * 2, hp.hidden_size)
        self.cbhg = CBHG(hidden_size // 2)

    def forward(self, input_):
        ##############################
        # input_: (batch, seq_length)
        ##############################
        # print('---------------------------------', np.shape(input_))  # [1, 25]
        # input_: (batch, length)
        # 转置矩阵，因为第一维是batch，所以转置的是第二维和第三维
        # print(np.shape(self.embed(input_)))
        # embedding后，变成了(batch, length, embedding)
        # 这里转置目的是为了适配prenet的大小
        input_ = torch.transpose(self.embed(input_), 1, 2)
        ##############################
        # input_: (batch, embedding, seq_length)
        ##############################
        # print('*************************', np.shape(input_))  # [1, 256, 25]
        prenet = self.prenet.forward(input_)
        # print(np.shape(prenet))
        ##############################
        # prenet: (batch, 128, seq_length)
        ##############################
        memory = self.cbhg.forward(prenet)
        # print(np.shape(memory))
        return memory
        ##############################
        # memory: (batch, seq_length, features=256)
        ##############################


class MelDecoder(nn.Module):
    """
    Decoder
    """

    def __init__(self):
        super(MelDecoder, self).__init__()
        self.prenet = Prenet(hp.num_mels, hp.hidden_size * 2, hp.hidden_size)
        self.attn_decoder = AttentionDecoder(hp.hidden_size * 2)

    def forward(self, decoder_input, memory):
        ##############################
        # memory: (batch, seq_length, features)
        ##############################

        ##############################
        # decoder_input: (batch, 80, seq_length)
        ##############################

        # Initialize hidden state of GRUcells
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(
            decoder_input.size()[0])
        outputs = list()
        # print(np.shape(decoder_input))

        # Training phase
        if self.training:
            # Prenet
            dec_input = self.prenet.forward(decoder_input)
            # print(np.shape(dec_input))
            timesteps = dec_input.size()[2] // hp.outputs_per_step

            # [GO] Frame
            prev_output = dec_input[:, :, 0]
            # print(np.shape(prev_output))

            for i in range(timesteps):
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(
                    prev_output, memory, attn_hidden=attn_hidden, gru1_hidden=gru1_hidden, gru2_hidden=gru2_hidden)

                outputs.append(prev_output)

                # print(random.random())
                if random.random() < hp.teacher_forcing_ratio:
                    # Get spectrum at rth position
                    prev_output = dec_input[:, :, i * hp.outputs_per_step]
                else:
                    # Get last output
                    prev_output = prev_output[:, :, -1]

            # Concatenate all mel spectrogram
            outputs = torch.cat(outputs, 2)

        else:
            # [GO] Frame
            prev_output = decoder_input

            for i in range(hp.max_iters):
                prev_output = self.prenet.forward(prev_output)
                prev_output = prev_output[:, :, 0]
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(
                    prev_output, memory, attn_hidden=attn_hidden, gru1_hidden=gru1_hidden, gru2_hidden=gru2_hidden)
                outputs.append(prev_output)
                prev_output = prev_output[:, :, -1].unsqueeze(2)

            outputs = torch.cat(outputs, 2)

        # print(np.shape(outputs))
        ##############################
        # outputs: (batch, 80, seq_length)
        ##############################
        return outputs


class PostProcessingNet(nn.Module):
    """
    Post-processing Network
    """

    def __init__(self):
        super(PostProcessingNet, self).__init__()
        self.postcbhg = CBHG(hp.hidden_size, K=8,
                             projection_size=hp.num_mels, is_post=True)
        self.linear = SeqLinear(hp.hidden_size * 2, hp.num_freq)

    def forward(self, input_):
        out = self.postcbhg.forward(input_)
        out = self.linear.forward(torch.transpose(out, 1, 2))

        return out


class Tacotron(nn.Module):
    """
    End-to-end Tacotron Network
    """

    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.embedding_size - hp.z_dim)
        self.vae = VAE()
        self.decoder = MelDecoder()
        self.postnet = PostProcessingNet()

    # def copy_z(self, z, copy_size):
    #     z_copied = torch.stack(
    #         [torch.stack([z[batch] for _ in range(copy_size)]) for batch in range(z.size(0))])
    #     # for batch in range(z.size(0)):
    #     #     torch.stack([z[batch] for _ in range(copy_size)])
    #     # print(z_copied)
    #     return z_copied

    # @torchsnooper.snoop()
    def forward(self, characters, mel_input):
        # print('-----------------------------------------', np.shape(mel_input))  # [1, 80, 1]
        z, mu, log_var = self.vae(mel_input)
        # print(np.shape(mu))
        # print(np.shape(log_var))

        # print(np.shape(mel_input))
        # print('----------------------', np.shape(z))
        # print('-----------------------------------------', np.shape(characters))  # [1, 25]
        memory = self.encoder.forward(characters)
        ##############################
        # memory: (batch, seq_length, 256)
        ##############################
        # print(np.shape(self.copy_z(z, memory.size(1))))
        # print('-----------------------------------------', np.shape(z))  # [32, 16]
        # print('-----------------------------------------', np.shape(memory))  # [1, 25, 240]
        # z = self.copy_z(z, memory.size(1))
        z = torch.stack([torch.stack([z[batch] for _ in range(
            memory.size(1))]) for batch in range(z.size(0))]).cuda()
        # print('-----------------------------------------', np.shape(z))  # [32, 25, 16]
        # print(memory)
        # memory = memory + z
        # print('-----------------------------------------', np.shape(memory))  # [1, 25, 240]
        memory = torch.cat((memory, z), 2)

        # print(np.shape(memory))
        # print(memory)
        # print(z)
        mel_output = self.decoder.forward(mel_input, memory)
        linear_output = self.postnet.forward(mel_output)

        return mel_output, linear_output, mu, log_var
