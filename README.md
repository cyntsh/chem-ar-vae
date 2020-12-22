# Reproducing and adapting results from Pati et al. (2020) to the chemical space
## About
Adds a loss term that is computed as the difference between the distance matrix for latent vectors at regularized dimension and the distance matrix for the desired attribute (in this case logP).

Code is adapted from the following repositories:
* [SELFIES](https://github.com/aspuru-guzik-group/selfies)
* [AR-VAE: Attribute-based Regularization of VAE Latent Spaces](https://github.com/ashispati/ar-vae)

## Results
The model seems to think that higher logP values are correlated with more carbon atoms, and lower logP values are correlated with more N atoms and double bonds. Some issues with disentanglement: dimension 0 is most correlated with dimensions 8 and 32, both of which seem to encode attributes relating to N, = vs C.

## References
* Ashis Pati, Alexander Lerch. "Attribute-based Regularization of Latent Spaces for Variational Auto-Encoders", Neural Computing and Applications. 2020.
* Mario Krenn, Florian Hase, AkshatKumar Nigam, Pascal Friederich, and Alan Aspuru-Guzik. Self-referencing embedded strings (selfies): A 100% robust molecular string representation. Machine Learning: Science and Technology, 1(3):045024, 2020.
