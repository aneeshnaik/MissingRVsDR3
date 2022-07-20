SELECT g.source_id, g.radial_velocity
FROM gaiadr3.gaia_source_lite AS g
WHERE g.radial_velocity IS NOT NULL
