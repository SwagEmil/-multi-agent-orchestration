#!/usr/bin/env python3
"""
Premium RAG Vector Visualization Script
Generates a high-end 3D interactive visualization using Three.js and WebGL.
Features: Glowing particles, constellation connections, glassmorphism UI.
"""

import sys
import os
from pathlib import Path
import json
import chromadb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def generate_premium_visualization():
    print("=" * 70)
    print("✨ GENERATING PREMIUM 3D RAG VISUALIZATION (Three.js)")
    print("=" * 70)
    
    # 1. Connect to Database
    db_path = Path(__file__).parent.parent / "data" / "vector_db"
    print(f"🔌 Connecting to ChromaDB at: {db_path}")
    
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection("ai_agent_knowledge_base")
        
        # 2. Fetch Data
        print("📥 Fetching embeddings and metadata...")
        data = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        embeddings = data['embeddings']
        documents = data['documents']
        metadatas = data['metadatas']
        ids = data['ids']
        
        count = len(embeddings)
        print(f"   Found {count} chunks")
        
        if count < 3:
            print("❌ Not enough data points (need at least 3)")
            return

        # 3. Analyze Structure (Clustering & Manifold Learning)
        print("🧠 Analyzing Semantic Structure...")
        X = np.array(embeddings)
        
        # A. Clustering (Find the "Topics")
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
        
        n_clusters = 6
        print(f"   Grouping into {n_clusters} semantic topics (K-Means)...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # B. Dimensionality Reduction (t-SNE is better for visualizing clusters than PCA)
        print("   Unfolding the manifold (384D -> 3D) using t-SNE...")
        # We use a higher perplexity to preserve global structure
        tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        # Normalize to range [-40, 40] for Three.js
        scaler = MinMaxScaler(feature_range=(-40, 40))
        X_norm = scaler.fit_transform(X_embedded)
        
        # 4. Prepare Data
        print("📝 Preparing WebGL data...")
        
        points_data = []
        
        # Generate distinct colors for CLUSTERS (Topics), not Sources
        # This shows the "Brain's Organization" better
        for i in range(count):
            source = metadatas[i].get('source', 'Unknown')
            chunk_id = metadatas[i].get('chunk_id', '?')
            doc_preview = documents[i][:300].replace('"', '&quot;').replace('\n', '<br>')
            
            # Color by Cluster (Topic)
            cluster_id = int(clusters[i])
            hue = cluster_id / n_clusters
            
            points_data.append({
                'x': float(X_norm[i, 0]),
                'y': float(X_norm[i, 1]),
                'z': float(X_norm[i, 2]),
                'source': source,
                'cluster': cluster_id,
                'hue': hue,
                'chunk_id': chunk_id,
                'preview': doc_preview
            })
        
        json_data = json.dumps(points_data)
        
        # 5. Generate HTML with Three.js
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Agent Knowledge Universe</title>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #050505; font-family: 'Inter', sans-serif; }}
        #canvas-container {{ width: 100vw; height: 100vh; }}
        
        /* Glassmorphism UI */
        #ui-layer {{
            position: absolute;
            top: 20px;
            left: 20px;
            pointer-events: none;
            z-index: 10;
        }}
        
        .card {{
            background: rgba(20, 20, 25, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            color: #fff;
            max-width: 350px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s ease;
            pointer-events: auto;
            margin-bottom: 15px;
        }}
        
        h1 {{ margin: 0 0 10px 0; font-size: 1.2rem; font-weight: 600; letter-spacing: -0.5px; background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        h2 {{ margin: 0 0 5px 0; font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
        p {{ margin: 0; font-size: 0.9rem; line-height: 1.5; color: #ccc; }}
        
        .stat-row {{ display: flex; justify-content: space-between; margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px; }}
        .stat {{ text-align: center; }}
        .stat-val {{ display: block; font-size: 1.2rem; font-weight: 700; }}
        .stat-label {{ font-size: 0.7rem; color: #666; }}
        
        #tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #333;
            color: #fff;
            padding: 10px;
            border-radius: 8px;
            pointer-events: none;
            display: none;
            max-width: 300px;
            font-size: 0.85rem;
            z-index: 20;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        
        .source-tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            background: rgba(255,255,255,0.1);
            margin-bottom: 8px;
        }}

        #loading {{
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            color: #4facfe; font-size: 1.5rem; font-weight: bold;
            text-shadow: 0 0 20px rgba(79, 172, 254, 0.5);
            transition: opacity 0.5s;
        }}
    </style>
    
    <!-- Three.js Import Map -->
    <script type="importmap">
        {{
            "imports": {{
                "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
            }}
        }}
    </script>
</head>
<body>
    <div id="loading">INITIALIZING NEURAL LINK...</div>
    <div id="canvas-container"></div>
    
    <div id="ui-layer">
        <div class="card">
            <h1>Agent Knowledge Universe</h1>
            <p>Interactive 3D visualization of the RAG vector space. Each particle represents a knowledge chunk.</p>
            
            <div class="stat-row">
                <div class="stat">
                    <span class="stat-val">{count}</span>
                    <span class="stat-label">VECTORS</span>
                </div>
                <div class="stat">
                    <span class="stat-val">{n_clusters}</span>
                    <span class="stat-label">TOPICS</span>
                </div>
                <div class="stat">
                    <span class="stat-val">t-SNE</span>
                    <span class="stat-label">LAYOUT</span>
                </div>
            </div>
        </div>
        
        <div id="info-card" class="card" style="opacity: 0; transform: translateY(20px);">
            <h2>SELECTED CHUNK</h2>
            <div id="chunk-source" class="source-tag">Source</div>
            <p id="chunk-content">Hover over a node...</p>
        </div>
    </div>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        import {{ EffectComposer }} from 'three/addons/postprocessing/EffectComposer.js';
        import {{ RenderPass }} from 'three/addons/postprocessing/RenderPass.js';
        import {{ UnrealBloomPass }} from 'three/addons/postprocessing/UnrealBloomPass.js';

        const data = {json_data};
        
        // Scene Setup
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x050505, 0.015);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(60, 40, 60);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.toneMapping = THREE.ReinhardToneMapping;
        document.getElementById('canvas-container').appendChild(renderer.domElement);
        
        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        
        // Post Processing (Bloom)
        const renderScene = new RenderPass(scene, camera);
        const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
        bloomPass.threshold = 0;
        bloomPass.strength = 1.2; // Glow strength
        bloomPass.radius = 0.5;
        
        const composer = new EffectComposer(renderer);
        composer.addPass(renderScene);
        composer.addPass(bloomPass);
        
        // Particles
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const sizes = [];
        
        const colorObj = new THREE.Color();
        
        data.forEach(point => {{
            positions.push(point.x, point.y, point.z);
            
            // Color based on source (HSL)
            colorObj.setHSL(point.hue, 0.8, 0.6);
            colors.push(colorObj.r, colorObj.g, colorObj.b);
            
            sizes.push(1.5);
        }});
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
        
        // Custom Shader Material for glowing dots
        const material = new THREE.PointsMaterial({{
            size: 1.5,
            vertexColors: true,
            map: createCircleTexture(),
            transparent: true,
            opacity: 0.9,
            sizeAttenuation: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        }});
        
        const particles = new THREE.Points(geometry, material);
        scene.add(particles);
        
        // Connections (Constellation effect)
        // Find close neighbors and draw lines (expensive O(N^2), so limit it)
        const lineMaterial = new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.08 }});
        const lineGeometry = new THREE.BufferGeometry();
        const linePositions = [];
        
        // Limit connections to avoid performance hit
        const maxConnections = 2000; 
        let connectionCount = 0;
        
        for (let i = 0; i < data.length; i++) {{
            for (let j = i + 1; j < data.length; j++) {{
                const dx = data[i].x - data[j].x;
                const dy = data[i].y - data[j].y;
                const dz = data[i].z - data[j].z;
                const distSq = dx*dx + dy*dy + dz*dz;
                
                if (distSq < 25) {{ // Threshold for connection
                    linePositions.push(data[i].x, data[i].y, data[i].z);
                    linePositions.push(data[j].x, data[j].y, data[j].z);
                    connectionCount++;
                    if (connectionCount > maxConnections) break;
                }}
            }}
            if (connectionCount > maxConnections) break;
        }}
        
        lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
        scene.add(lines);
        
        // Interaction
        const raycaster = new THREE.Raycaster();
        raycaster.params.Points.threshold = 1.0;
        const mouse = new THREE.Vector2();
        
        // Hover state
        let hoveredIndex = -1;
        const highlightGeometry = new THREE.BufferGeometry();
        highlightGeometry.setAttribute('position', new THREE.Float32BufferAttribute([0,0,0], 3));
        const highlightMaterial = new THREE.PointsMaterial({{ 
            color: 0xffffff, size: 4, sizeAttenuation: true, 
            map: createCircleTexture(), transparent: true, opacity: 1, blending: THREE.AdditiveBlending 
        }});
        const highlightPoint = new THREE.Points(highlightGeometry, highlightMaterial);
        highlightPoint.visible = false;
        scene.add(highlightPoint);
        
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('resize', onWindowResize);
        
        // Remove loading screen
        document.getElementById('loading').style.opacity = 0;
        setTimeout(() => document.getElementById('loading').remove(), 500);
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            composer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        function createCircleTexture() {{
            const canvas = document.createElement('canvas');
            canvas.width = 32; canvas.height = 32;
            const context = canvas.getContext('2d');
            const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
            gradient.addColorStop(0, 'rgba(255,255,255,1)');
            gradient.addColorStop(0.2, 'rgba(255,255,255,0.8)');
            gradient.addColorStop(0.5, 'rgba(255,255,255,0.2)');
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            context.fillStyle = gradient;
            context.fillRect(0, 0, 32, 32);
            const texture = new THREE.Texture(canvas);
            texture.needsUpdate = true;
            return texture;
        }}
        
        // Animation Loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            
            // Raycasting
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(particles);
            
            if (intersects.length > 0) {{
                const index = intersects[0].index;
                
                if (hoveredIndex !== index) {{
                    hoveredIndex = index;
                    document.body.style.cursor = 'pointer';
                    
                    // Highlight
                    const p = data[index];
                    highlightPoint.position.set(p.x, p.y, p.z);
                    highlightPoint.visible = true;
                    
                    // Update UI
                    const infoCard = document.getElementById('info-card');
                    infoCard.style.opacity = 1;
                    infoCard.style.transform = 'translateY(0)';
                    
                    document.getElementById('chunk-source').textContent = p.source;
                    document.getElementById('chunk-source').style.backgroundColor = `hsl(${{p.hue * 360}}, 70%, 30%)`;
                    document.getElementById('chunk-content').innerHTML = p.preview;
                }}
            }} else {{
                if (hoveredIndex !== -1) {{
                    hoveredIndex = -1;
                    document.body.style.cursor = 'default';
                    highlightPoint.visible = false;
                    
                    const infoCard = document.getElementById('info-card');
                    infoCard.style.opacity = 0;
                    infoCard.style.transform = 'translateY(20px)';
                }}
            }}
            
            // Subtle rotation of lines
            lines.rotation.y += 0.0005;
            
            composer.render();
        }}
        
        animate();
    </script>
</body>
</html>
        """
        
        output_path = Path(__file__).parent.parent / "rag_universe.html"
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        print(f"✅ Premium visualization saved to: {output_path}")
        print("🚀 Opening in browser...")
        
        # Try to open automatically
        try:
            if sys.platform == 'darwin':  # macOS
                os.system(f'open "{output_path}"')
            elif sys.platform == 'linux':
                os.system(f'xdg-open "{output_path}"')
            elif sys.platform == 'win32':
                os.system(f'start "{output_path}"')
        except:
            pass

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    generate_premium_visualization()
