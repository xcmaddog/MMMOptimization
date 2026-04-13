import numpy as np
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time

def build_ephemeris_cache(start_date="2024-01-01", end_date="2032-01-01", step_days=0.25, filename="ephemeris_cache.npz"):
    """
    Queries Astropy for states at a higher resolution (default 6 hours).
    """
    t_start = Time(start_date, scale="utc")
    t_end = Time(end_date, scale="utc")
    
    total_days = (t_end - t_start).jd
    # Generate steps (e.g., 0, 0.25, 0.5...)
    time_steps = np.arange(0, total_days + step_days, step_days)
    sample_times = t_start + time_steps * u.day
    
    print(f"Generating {len(sample_times)} points ({step_days} day steps)...")
    
    with solar_system_ephemeris.set("builtin"):
        sun_pos, sun_vel = get_body_barycentric_posvel("sun", sample_times)
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", sample_times)
        mars_pos, mars_vel = get_body_barycentric_posvel("mars", sample_times)
        
    # Standardize to Heliocentric KM and KM/S
    earth_r = (earth_pos.xyz - sun_pos.xyz).to_value(u.km).T
    earth_v = (earth_vel.xyz - sun_vel.xyz).to_value(u.km / u.s).T
    mars_r = (mars_pos.xyz - sun_pos.xyz).to_value(u.km).T
    mars_v = (mars_vel.xyz - sun_vel.xyz).to_value(u.km / u.s).T
    
    np.savez_compressed(
        filename,
        jd=sample_times.jd,
        earth_r=earth_r,
        earth_v=earth_v,
        mars_r=mars_r,
        mars_v=mars_v
    )
    print(f"Cache complete: {filename}")

if __name__ == "__main__":
    build_ephemeris_cache()