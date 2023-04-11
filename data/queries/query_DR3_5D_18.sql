SELECT source_id, phot_g_mean_mag, nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved, rv_template_teff, radial_velocity, radial_velocity_error, parallax, parallax_error, ra, dec, pmra, pmdec, ra_error, ra_dec_corr, ra_pmra_corr, ra_pmdec_corr, dec_error, dec_pmra_corr, dec_pmdec_corr, pmra_error, pmra_pmdec_corr, pmdec_error
FROM gaiadr3.gaia_source
WHERE (radial_velocity IS NULL) AND (parallax IS NOT NULL) AND (pmra IS NOT NULL) AND (pmdec IS NOT NULL) AND (phot_g_mean_mag IS NOT NULL) AND (phot_g_mean_mag >= 16.8) AND (phot_g_mean_mag < 16.85)

