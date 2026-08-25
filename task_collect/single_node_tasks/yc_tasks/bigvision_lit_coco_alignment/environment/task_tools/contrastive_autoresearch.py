#!/usr/bin/env python3
"""Fixed trainer launcher that binds the candidate loss at one audited seam."""

from absl import app
from candidate import alignment_loss
from big_vision.trainers.proj.image_text import contrastive


def main(argv):
    contrastive.u.bidirectional_contrastive_loss = alignment_loss.bidirectional_contrastive_loss
    return contrastive.main(argv)


if __name__ == "__main__":
    app.run(main)

