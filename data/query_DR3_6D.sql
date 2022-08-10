SELECT g.source_id, g.phot_g_mean_mag, g.rv_template_teff, g.radial_velocity, g.radial_velocity_error, g.parallax, g.ra, g.dec, g.pmra, g.pmdec, g.parallax_error, g.ra_parallax_corr, g.dec_parallax_corr, g.parallax_pmra_corr, g.parallax_pmdec_corr, g.ra_error, g.ra_dec_corr, g.ra_pmra_corr, g.ra_pmdec_corr, g.dec_error, g.dec_pmra_corr, g.dec_pmdec_corr, g.pmra_error, g.pmra_pmdec_corr, g.pmdec_error
FROM gaiadr3.gaia_source AS g
WHERE g.radial_velocity IS NOT NULL
