def cond_fn_sdg(x, mu, t, y=None, 
                    ref_text=None, ref_text_length=None, 
                    ref_mel=None, ref_mel_length=None):

        with torch.no_grad():
            if text_weight != 0:
                assert y is not None
                text_features = te(y)
            if audio_weight != 0:
                ref_audio_noised, ref_mu, _, _ = generator.given_t_get_xt(ref_text, ref_text_length, 
                                                    ref_mel, ref_mel_length, spk=spk, t=t)
                ref_audio_noised = ref_audio_noised.unsqueeze(1).permute(0, 1, 3, 2)
                ref_mu           = ref_mu.unsqueeze(1).permute(0, 1, 3, 2)
                ref_audio_features, _ = ae(ref_audio_noised, ref_mu, t)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            x_in = x_in.unsqueeze(1).permute(0, 1, 3, 2)
            mu   = mu.unsqueeze(1).permute(0, 1, 3, 2)
            audio_features, _ = ae(x_in, mu, t)
            if text_weight != 0:
                loss_text = text_loss(audio_features, text_features)
            else:
                loss_text = 0
            if audio_weight != 0:
                loss_audio = audio_loss(audio_features, ref_audio_features, audio_loss_type)
            else:
                loss_audio = 0
            total_guidance = loss_text * text_weight + loss_audio * audio_weight

            return torch.autograd.grad(total_guidance.sum(), x_in)[0]