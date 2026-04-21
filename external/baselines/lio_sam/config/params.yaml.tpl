# SUP-01 Phase E: params template for JokerJohn/LIO_SAM_6AXIS community
# fork. Top-level namespace changed from `lio_sam:` (upstream TixiaoShan)
# to `lio_sam_6axis:`. Fork adds imu_type, debug*, GPS origin-refinement
# keys. Backup of upstream template in params.yaml.tpl.bak_upstream.

lio_sam_6axis:

  # Topics — match our rosbag topic names
  pointCloudTopic: "/velodyne_points"
  imuTopic: "/imu/data"
  odomTopic: "odometry/imu"
  gpsTopic: "odometry/gpsz"

  # Frames
  lidarFrame: "velodyne"
  baselinkFrame: "velodyne"
  odometryFrame: "odom"
  mapFrame: "map"

  # GPS (disabled for KITTI Odometry — no GPS topic on our bag).
  useGPS: false
  updateOrigin: false
  gpsFrequence: 10
  gpsDistance: 0.5
  # useImuHeadingInitialization = false: OxTS yaw is absolute ENU heading
  # (north = 0, CCW); LIO-SAM would otherwise rotate its first pose by
  # that angle, offsetting from KITTI GT's "first pose = identity"
  # convention by the vehicle's initial heading (~60 deg on Seq 00).
  # Empirically this is also needed to avoid IMU-preintegration runaway;
  # see SUP-01 P0-1 + sup-notes.md.
  useImuHeadingInitialization: false
  useGpsElevation: false
  gpsCovThreshold: 2.0
  poseCovThreshold: 25.0

  # 6AXIS debug / logging settings (fork additions, all false for batch run)
  debugLidarTimestamp: false
  debugImu: false
  debugGps: false

  # Save
  savePCD: false
  savePCDDirectory: "/tmp/lio_sam/"

  # Sensor: KITTI Velodyne HDL-64E
  sensor: velodyne
  N_SCAN: 64
  Horizon_SCAN: 1800
  downsampleRate: 1
  lidarMinRange: 5.0
  lidarMaxRange: 100.0

  # IMU Settings — OxTS RT3003 datasheet values. R1 loosening experiment
  # (100× σ) made upstream APE 6× worse (1076m → 6406m), confirming the
  # upstream P0-2 bug is not noise-tuning-related. 6AXIS fork is expected
  # to handle the IMU convention correctly; retaining datasheet sigmas.
  imuAccNoise: 3.9939570888238808e-03
  imuGyrNoise: 1.5636343949698187e-03
  imuAccBiasN: 6.4356659353532566e-05
  imuGyrBiasN: 3.5640318696367613e-05
  imuGravity: 9.80511
  imuRPYWeight: 0.01

  # 6AXIS-specific: IMU type selector. 0 = 6-axis (accel + gyro only,
  # no magnetometer), 1 = 9-axis. KITTI OxTS RT3003 publishes orientation
  # from a 9-axis suite but we don't feed the magnetometer; treating it as
  # 6-axis avoids the fork's magnetometer-based yaw refinement step.
  imu_type: 0

  # Extrinsics — RENDERED AT RUNTIME by envsubst (Stage C of SUP-01 P0-1).
  # run.sh derives R/T from KITTI Raw calib_imu_to_velo.txt for the current
  # sequence date and sets KITTI_EXT_TRANS / KITTI_EXT_ROT before launching.
  # One image slam-baselines/lio_sam:latest serves all sequences; the
  # per-sequence calibration is picked up from the $SEQ env var.
  # extrinsicRPY set equal to extrinsicRot; forcing identity regresses
  # upstream Seq 00 SE(3) APE by ~15 m.
  extrinsicTrans: [${KITTI_EXT_TRANS}]
  extrinsicRot: [${KITTI_EXT_ROT}]
  extrinsicRPY: [${KITTI_EXT_ROT}]

  # LOAM feature threshold
  edgeThreshold: 1.0
  surfThreshold: 0.1
  edgeFeatureMinValidNum: 10
  surfFeatureMinValidNum: 100

  # Voxel filter
  # SUP-01 Phase F.3 (2026-04-19): tested mappingSurfLeafSize 0.4 → 0.2
  # (finer surface voxel). Result APE_SE3 31.66 → 407.73 m (+1188%,
  # CATASTROPHIC), total path 17243 m (4.6x GT), RPE 28 m — scan-to-map
  # ICP collapses. Finer voxel over-fits per-scan noise; LIO-SAM's
  # feature-based mapping is tuned for 0.4 m surface voxels on KITTI
  # HDL-64E dense output. Reverted.
  odometrySurfLeafSize: 0.4
  mappingCornerLeafSize: 0.2
  mappingSurfLeafSize: 0.4

  # Motion constraint
  z_tollerance: 1000
  rotation_tollerance: 1000

  # CPU
  numberOfCores: 4
  mappingProcessInterval: 0.15

  # Surrounding map
  # SUP-01 Phase F.1 (2026-04-19): tested surroundingkeyframeAddingDistThreshold
  # 1.0 → 0.5 (halve keyframe distance). Result: APE_SE3 31.66 → 49.54 m
  # (+56%, REGRESSED). Denser keyframes harm Seq 00: more low-overlap
  # pairs swamp the factor graph with weak constraints. Reverted to 1.0.
  surroundingkeyframeAddingDistThreshold: 1.0
  surroundingkeyframeAddingAngleThreshold: 0.2
  surroundingKeyframeDensity: 2.0
  surroundingKeyframeSearchRadius: 50.0

  # Loop closure
  # SUP-01 Phase F.2 (2026-04-19): tested loopClosureFrequency 1→2 +
  # historyKeyframeSearchTimeDiff 30→15. Result APE_SE3 31.66 → 42.43 m
  # (+34%, REGRESSED). Faster loop closure + shorter time-diff introduces
  # false positives — near-time keyframes have too much visual similarity
  # from same vehicle heading, producing wrong alignments. Reverted.
  loopClosureEnableFlag: true
  loopClosureFrequency: 1.0
  surroundingKeyframeSize: 50
  historyKeyframeSearchRadius: 15.0
  historyKeyframeSearchTimeDiff: 30.0
  historyKeyframeSearchNum: 25
  historyKeyframeFitnessScore: 0.3

  # Visualization
  globalMapVisualizationSearchRadius: 1000.0
  globalMapVisualizationPoseDensity: 10.0
  globalMapVisualizationLeafSize: 1.0

  # 6AXIS-specific: output map voxel (saved by /lio_sam_6axis/save_map)
  globalMapLeafSize: 0.5
