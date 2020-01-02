fornax=coords.SkyCoord(ra='13h25m27.6s', dec='-43d01m11s')
pointing_separation = j2000_pointing.separation(fornax)

num_dates = dates.date2num(times.to_datetime())

plt.plot(num_dates, pointing_separation)
plt.show()
unix_times=rd.times
from astropy.time import Time
times = Time(unix_times, format='unix')

hartrao = coords.EarthLocation.from_geodetic(
    lon=27.6853931*units.deg,
    lat=-25.8897515*units.deg,
    height=1415.710*units.m,
    ellipsoid='WGS84')

alt, az = (90-17.1)*units.deg, 180*units.deg 
altazs = coords.SkyCoord(frame='altaz', alt=alt*np.ones(len(times)),az=az*np.ones(len(times)),
                         location=hartrao, obstime=times)
j2000_pointing = altazs.transform_to(coords.ICRS)

fornax = coords.SkyCoord(ra='03h22m41.7s', dec='-37d12m30s')
fornax=coords.SkyCoord(ra='13h25m27.6s', dec='-43d01m11s')
pointing_separation = j2000_pointing.separation(fornax)

num_dates = dates.date2num(times.to_datetime())

plt.plot(num_dates, pointing_separation)
plt.show()
unix_times = rd['index_map/time']['ctime']
from astropy.time import Time
times = Time(unix_times, format='unix')

hartrao = coords.EarthLocation.from_geodetic(
    lon=27.6853931*units.deg,
    lat=-25.8897515*units.deg,
    height=1415.710*units.m,
    ellipsoid='WGS84')

alt, az = (90-17.1)*units.deg, 180*units.deg 
altazs = coords.SkyCoord(frame='altaz', alt=alt*np.ones(len(times)),az=az*np.ones(len(times)),
                         location=hartrao, obstime=times)
j2000_pointing = altazs.transform_to(coords.ICRS)

fornax = coords.SkyCoord(ra='03h22m41.7s', dec='-37d12m30s')
fornax=coords.SkyCoord(ra='13h25m27.6s', dec='-43d01m11s')
pointing_separation = j2000_pointing.separation(fornax)

num_dates = dates.date2num(times.to_datetime())

plt.plot(num_dates, pointing_separation)
plt.show()
unix_times=rd.times
from astropy.time import Time
times = Time(unix_times, format='unix')

hartrao = coords.EarthLocation.from_geodetic(
    lon=27.6853931*units.deg,
    lat=-25.8897515*units.deg,
    height=1415.710*units.m,
    ellipsoid='WGS84')

alt, az = (90-10.2)*units.deg, 180*units.deg 
altazs = coords.SkyCoord(frame='altaz', alt=alt*np.ones(len(times)),az=az*np.ones(len(times)),
                         location=hartrao, obstime=times)
j2000_pointing = altazs.transform_to(coords.ICRS)

fornax = coords.SkyCoord(ra='03h22m41.7s', dec='-37d12m30s')
fornax=coords.SkyCoord(ra='13h25m27.6s', dec='-43d01m11s')
pointing_separation = j2000_pointing.separation(fornax)

num_dates = dates.date2num(times.to_datetime())

plt.plot(num_dates, pointing_separation)
plt.show()
unix_times=rd.times
from astropy.time import Time
times = Time(unix_times, format='unix')

hartrao = coords.EarthLocation.from_geodetic(
    lon=27.6853931*units.deg,
    lat=-25.8897515*units.deg,
    height=1415.710*units.m,
    ellipsoid='WGS84')

alt, az = (90-17.1)*units.deg, 180*units.deg 
altazs = coords.SkyCoord(frame='altaz', alt=alt*np.ones(len(times)),az=az*np.ones(len(times)),
                         location=hartrao, obstime=times)
j2000_pointing = altazs.transform_to(coords.ICRS)

fornax = coords.SkyCoord(ra='03h22m41.7s', dec='-37d12m30s')
fornax=coords.SkyCoord(ra='13h25m27.6s', dec='-43d01m11s')
pointing_separation = j2000_pointing.separation(fornax)

num_dates = dates.date2num(times.to_datetime())

plt.plot(num_dates, pointing_separation)
plt.show()
plt.plot(pointing_separation)
plt.show()

## ---(Wed Mar 13 13:47:22 2019)---
runfile('C:/Users/zahra/.spyder-py3/systemp.py', wdir='C:/Users/zahra/.spyder-py3')
r=[100,200,300,400,500,590,600,700,800]

for i in r:
    band_ind=i
    filtered = rd.filtered_on_time((3, 3), filter_type='low', char_freqs=1/0.5)
    y=np.abs(filtered[:, band_ind])
    plt.plot(y)
    plt.show()

runfile('C:/Users/zahra/.spyder-py3/systemp.py', wdir='C:/Users/zahra/.spyder-py3')
plt.plot(pointing(rd.times, 79.8, 180, frame='Galactic').b)
runfile('C:/Users/zahra/.spyder-py3/systemp.py', wdir='C:/Users/zahra/.spyder-py3')
import imfit