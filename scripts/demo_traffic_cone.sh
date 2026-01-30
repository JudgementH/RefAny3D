python demo.py \
    --prompt "Photorealistic image of an upright orange traffic cone with two white reflective stripes, weathered plastic, and a circular base resting on a dark platform; set against a softly blurred urban construction-scene backdrop—chain-link fencing, yellow caution tape, distant cranes, asphalt, and faint city silhouettes—lit by warm golden-hour light from the left. Keep the cone crisp with shallow depth of field so the background provides context without competing for attention. Use a harmonious color palette of orange, industrial gray, and pale blue, with long soft shadows and a gentle rim highlight along the cone. Camera at eye level with a slight tilt to convey depth; diffusion-friendly rendering with fine texture, scratches, dirt, and subtle film grain to enhance realism." \
    --glb_path "./data/traffic_cone/model/traffic_cone.glb" \
    --output_dir "./data/traffic_cone/output" \
    --base_model "/home/gentoo/docker_shared/asus/huanghz/weight/FLUX.1-dev" 
