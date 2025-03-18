"""
Perceptual Similarity Metric

https://github.com/richzhang/PerceptualSimilarity

Usage:

.. code-block:: python

   from olimp.evaluation.loss.lpips import LPIPS

   # best forward scores
   loss_fn_alex = LPIPS(net='alex')

   # closer to "traditional" perceptual loss, when used for optimization
   loss_fn_vgg = LPIPS(net='vgg')
"""

from lpips import LPIPS as LPIPS
