import torch
from packaging import version


class PatchNCELoss:
    """
    Location Contrastive loss
    """

    def __init__(self, batch_size=16, nce_t=0.07):
        super().__init__()
        self.batch_size = batch_size
        self.nce_t = nce_t
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = (
            torch.uint8
            if version.parse(torch.__version__) < version.parse("1.2.0")
            else torch.bool
        )

    # feat_q and feat_k: 256 x 256, dim0: num_patches, dim1: feater length
    def __call__(self, feat_q, feat_k):
        batch_size = feat_q.shape[0]
        emb_dims = feat_q.shape[1]  # 256
        chs = feat_q.shape[2]  # 256
        feat_k = feat_k.detach()

        # feat_q and feat_k: 256 x 256, dim0: num_patches, dim1: feater length
        # pos logit, the corresponding patches in each batch
        l_pos = torch.zeros([batch_size, emb_dims, 1], device=feat_q.device)
        for i in range(batch_size):
            l_pos[i, ...] = torch.bmm(
                feat_q[i, ...].view(emb_dims, 1, -1),
                feat_k[i, ...].view(emb_dims, -1, 1),
            ).squeeze(2)
        # 256 x 1 x 1 -> 256 x 1
        # l_pos = l_pos.view(emb_dims, 1)

        # neg logit, cosine similarity of all the patches that are not corresponding

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.

        # reshape features to batch size, feat_q and feat_k: 1 x 256 x 256
        l_neg_curbatch = torch.bmm(
            feat_q.view(batch_size, -1, chs), feat_k.view(batch_size, chs, -1)
        )

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(emb_dims, device=feat_q.device, dtype=self.mask_dtype)[
            None, :, :
        ]  # 1 x 256 x 256
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(batch_size, -1, emb_dims)  # 256 x 256

        out = torch.cat((l_pos, l_neg), dim=2) / self.nce_t
        loss = torch.zeros([out.shape[0]]).to(device=feat_q.device)
        for i in range(out.shape[0]):
            loss[i] = self.cross_entropy_loss(
                out[i],
                torch.zeros(out[i].size(0), dtype=torch.long, device=feat_q.device),
            ).mean()

        return loss


class DisNCELoss:
    """
    Layer Contrastive loss
    """

    def __init__(
        self, *, num_patches_pos: int = 8, num_patches_neg: int = 128, nce_t=0.07
    ):
        super().__init__()
        self.num_patches_pos = num_patches_pos
        self.num_patches_neg = num_patches_neg
        self.nce_t = nce_t

    # feat_B for the background, feat_R for the rain
    # shape: (num_patches * batch_size, feature length)
    def __call__(self, featB, featR):
        batch_size = featB.shape[1] // self.num_patches_pos
        # if featR.shape[0] != num_patches*batch_size:
        #     raise ValueError('num_patches of rain and background are not equal')

        # making labels
        labels = torch.cat(
            [torch.ones(self.num_patches_pos, 1), torch.zeros(self.num_patches_neg, 1)],
            dim=0,
        )

        loss_dis_total = torch.zeros([featB.shape[0]], device=featB.device)
        # obtain each background and the rain layer to calculate the disentangle loss
        for batch in range(featB.shape[0]):
            for i in range(batch_size):
                cur_featB = featB[
                    batch, i * self.num_patches_pos : (i + 1) * self.num_patches_pos, :
                ]
                cur_featR = featR[
                    batch, i * self.num_patches_neg : (i + 1) * self.num_patches_neg, :
                ]
                cur_disloss = self.cal_each_disloss(cur_featB, cur_featR, labels)
                loss_dis_total[batch] += cur_disloss
        return loss_dis_total

    # cur_featB: [num_patches, feature length]
    # labels: [num_patches*2, 1]
    @torch.no_grad()
    def cal_each_disloss(self, cur_featB, cur_featR, labels):
        featFusion = torch.cat([cur_featB, cur_featR], dim=0)
        mask = torch.eq(labels, labels.t()).float().to(cur_featB.device)

        num_patches = featFusion.shape[0]
        # contrast_count: number of all the rain and background patches
        contrast_feature = featFusion
        #         contrast_count = featFusion.shape[1]
        contrast_count = 1
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits of all the elements
        # Denoting: zi: one sample, zl: all the other samples, zp: positives to zi, za: negatives to zi
        # anchor_dot_contrast = zi * zl
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.t()), self.nce_t
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask, repeat the masks to match the n_views of positives
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.ones_like(mask).scatter_(
            1,
            torch.arange(num_patches * anchor_count).view(-1, 1).to(cur_featB.device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        # exp_logits: exp(zi * zl)
        exp_logits = torch.exp(logits) * logits_mask
        # the meaning of sum(1): sum the cosine similarity of one sample and all the other samples
        # log_prob: (zi*zl) - log(sum(exp(zi,zl))) = log[exp(zi*zl) / sum(exp(zi * zl)) ]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # (mask * log_prob).sum(1): log [sum(exp(zi*zp)) / sum(exp(zi*zl)) ]
        # mask.sum(1): |P(i)|
        # mean_log_prob_pos: L^sup_out
        loss = -(mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = loss.view(anchor_count, num_patches).mean()

        return loss
