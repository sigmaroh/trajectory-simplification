# 3. Dataset Analysis

## 3.1 GeoLife GPS Dataset

The GeoLife GPS trajectory dataset [Zheng et al., 2009] is a publicly available dataset containing GPS trajectories collected from 182 users over a period of more than five years (from April 2007 to August 2012). The dataset includes 17,621 trajectories with a total distance of about 1.2 million kilometers and a total duration of 48,000+ hours.

### 3.1.1 Dataset Characteristics

- **Users**: 182 individuals
- **Trajectories**: 17,621 files
- **Total Points**: Over 24 million GPS points
- **Geographic Coverage**: Primarily Beijing, China, with some trajectories in other cities
- **Sampling Rate**: Variable, typically 1-5 seconds, but highly irregular
- **Transportation Modes**: Walking, biking, bus, car, subway, train (airplane trips excluded — see Section 3.2)

### 3.1.2 Data Format

Each trajectory is stored in a `.plt` file with the following format:
- Columns: latitude, longitude, 0, altitude, date (YYYY-MM-DD), time (HH:MM:SS)
- First 6 lines contain metadata (user ID, trajectory ID, etc.)

## 3.2 Preprocessing

### 3.2.1 Data Cleaning

We applied the following preprocessing steps:

1. **Airplane Trajectory Exclusion**: Airplane trips are fundamentally different from ground mobility (long-distance, straight-line, non-urban) and are excluded before any other filtering. Two complementary methods are used:
   - **Label-based (primary)**: For the 69 users who provided transportation mode labels (`labels.txt`), any `.plt` file whose time window overlaps with a segment labeled `airplane` is discarded.
   - **Speed-based heuristic (fallback)**: For users without label files, trajectories whose mean speed exceeds 80 m/s (≈ 288 km/h — above any realistic ground vehicle) are discarded. This threshold is deliberately set below aircraft takeoff speed (~70–80 m/s) to catch low-altitude flight phases that a point-level speed filter would miss.
2. **Outlier Removal**: Removed individual GPS points with unrealistic speeds (>200 m/s ≈ 720 km/h), which indicate GPS measurement errors.
3. **Spatial Outlier Detection**: Removed points with very large spatial jumps using Median Absolute Deviation (MAD) with a 5-MAD threshold.
4. **Duplicate Removal**: Removed duplicate timestamps.
5. **Minimum Length Filter**: Kept only trajectories with at least 50 points after cleaning.

### 3.2.2 Trajectory Properties Computation

For each trajectory, we computed:

- **Sampling Intervals**: Time differences between consecutive points
- **Sampling Irregularity**: Coefficient of variation (CV) of sampling intervals
- **Speeds**: Computed using Haversine distance and time intervals
- **Direction Changes**: Angular differences between consecutive segments
- **Stops**: Points with speed < 1 m/s (≈ 3.6 km/h)
- **Turns**: Points with direction change > 30 degrees

## 3.3 Dataset Statistics

### 3.3.1 Trajectory Length Distribution

Our preprocessed dataset contains X trajectories with the following length distribution:

- **Mean Points**: X points per trajectory
- **Median Points**: X points
- **Min/Max Points**: X / X points
- **Standard Deviation**: X points

The distribution shows [describe distribution: normal, skewed, etc.]

### 3.3.2 Sampling Irregularity

The coefficient of variation (CV) of sampling intervals measures irregularity:

- **Mean CV**: X (where CV = std/mean)
- **High Irregularity (CV > 1.0)**: X% of trajectories
- **Regular Sampling (CV < 0.3)**: X% of trajectories

This indicates that [interpretation: most trajectories have irregular sampling, etc.]

### 3.3.3 Speed Distribution

- **Mean Speed**: X m/s (≈ X km/h)
- **Max Speed**: X m/s (≈ X km/h)
- **Stop Ratio**: X% of points are stops (speed < 1 m/s)

### 3.3.4 Turn Distribution

- **Mean Turn Ratio**: X% of points are turns (direction change > 30°)
- **Trajectories with Turns**: X% contain at least one turn

### 3.3.5 Duration and Distance

- **Mean Duration**: X seconds (≈ X minutes)
- **Mean Distance**: X meters (≈ X km)
- **Total Distance**: X km across all trajectories

## 3.4 Challenges Identified

The dataset analysis reveals several challenges for trajectory simplification:

1. **High Irregularity**: Most trajectories exhibit significant sampling irregularity (CV > 0.5), making uniform sampling assumptions invalid.

2. **Noise**: GPS measurements contain noise, evident in small random variations in coordinates even during stops.

3. **Feature Diversity**: Trajectories vary widely in terms of stops, turns, and speed patterns, requiring adaptive methods.

4. **Scale Variation**: Trajectory lengths range from tens to thousands of points, requiring scalable algorithms.

## 3.5 Representative Examples

[Include visualizations and descriptions of representative trajectories showing different characteristics: high irregularity, many stops, many turns, etc.]

