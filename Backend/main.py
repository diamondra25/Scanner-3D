from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer
from fastapi import WebSocket
import os
import cv2
import numpy as np
import json
import asyncio
import aiofiles
from typing import List, Dict, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import uuid
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
import hashlib
from pathlib import Path
import time
from dataclasses import dataclass
import gzip
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import shutil
from slowapi import Limiter
from slowapi.util import get_remote_address
from contextlib import contextmanager


limiter = Limiter(key_func=get_remote_address)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Modèles Pydantic
class ScanStatus(str, Enum):
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class QualityEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class StartScanRequest(BaseModel):
    name: Optional[str] = Field(None, description="Nom de la session")
    timestamp: Optional[str] = Field(None, description="Timestamp de création")

class ProcessScanRequest(BaseModel):
    timestamp: Optional[str] = Field(None, description="Timestamp de traitement")
    quality: QualityEnum = Field(QualityEnum.MEDIUM, description="Qualité du traitement: low, medium, high")

@dataclass
class ScanImage:
    filename: str
    path: str
    uploaded_at: str
    file_size: int
    checksum: str

@dataclass
class ScanSession:
    id: str
    name: str
    status: ScanStatus
    created_at: str
    updated_at: str
    images: List[ScanImage]
    progress: float
    model_path: Optional[str] = None
    model_filename: Optional[str] = None
    error_message: Optional[str] = None
    processing_start_time: Optional[str] = None
    processing_end_time: Optional[str] = None
    total_processing_time: Optional[float] = None

@dataclass
class CameraParameters:
    """Paramètres intrinsèques de la caméra"""
    fx: float  # Focale en x
    fy: float  # Focale en y
    cx: float  # Centre principal x
    cy: float  # Centre principal y
    k1: float = 0.0  # Distorsion radiale
    k2: float = 0.0
    p1: float = 0.0  # Distorsion tangentielle
    p2: float = 0.0
    
    def to_matrix(self) -> np.ndarray:
        """Convertir en matrice intrinsèque"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

# Configuration
CONFIG = {
    "MAX_UPLOAD_SIZE": 10 * 1024 * 1024,  # 10MB
    "MAX_IMAGES_PER_SESSION": 100,
    "SUPPORTED_FORMATS": [".jpg", ".jpeg", ".png"],
    "SESSION_TIMEOUT": 3600,  # 1 heure
    "CLEANUP_INTERVAL": 300,  # 5 minutes
    "MAX_CONCURRENT_PROCESSING": 3,
    "FEATURE_EXTRACTION_TIMEOUT": 30,
    "POINT_CLOUD_GENERATION_TIMEOUT": 60,
    "MAX_WORKERS": max(4, os.cpu_count() or 4)
}

async def save_compressed_model(model_data: dict, output_path: str):
    """Sauvegarder le modèle compressé"""
    temp_path = output_path + '.tmp'
    compressed_path = output_path + '.gz'
    
    await save_point_cloud_ply(model_data, temp_path)
    
    with open(temp_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    os.remove(temp_path)
    return compressed_path

# Nettoyage automatique
async def cleanup_old_files(max_age_days: int = 7):
    """Nettoyer les fichiers anciens"""
    import time
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    for directory in [UPLOAD_DIR, MODELS_DIR, THUMBNAILS_DIR]:
        for file_path in Path(directory).glob('*'):
            if current_time - file_path.stat().st_mtime > max_age_seconds:
                file_path.unlink()

# Dossiers pour les données
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
SCANS_DIR = BASE_DIR / "scans"
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"
THUMBNAILS_DIR = BASE_DIR / "thumbnails"

# Créer les dossiers
for directory in [UPLOAD_DIR, SCANS_DIR, MODELS_DIR, TEMP_DIR, THUMBNAILS_DIR]:
    directory.mkdir(exist_ok=True)

@dataclass
class ProcessingMetrics:
    start_time: float
    feature_extraction_time: float
    point_cloud_generation_time: float
    file_save_time: float
    total_time: float
    memory_usage: Dict[str, Any]

# Logging structuré
import structlog
logger = structlog.get_logger()

# Dans process_scan_background
metrics = ProcessingMetrics(
    start_time=time.time(),
    feature_extraction_time=0.0,
    point_cloud_generation_time=0.0,
    file_save_time=0.0,
    total_time=0.0,
    memory_usage=0.0
)


# Stockage en mémoire avec thread-safety
class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, ScanSession] = {}
        self._lock = threading.RLock()
        self._processing_semaphore = threading.Semaphore(CONFIG["MAX_CONCURRENT_PROCESSING"])
    
    def create_session(self, name: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        session = ScanSession(
            id=session_id,
            name=name or f"Scan {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            status=ScanStatus.ACTIVE,
            created_at=current_time,
            updated_at=current_time,
            images=[],
            progress=0.0
        )
        
        with self._lock:
            self._sessions[session_id] = session
        
        logger.info(f"Session créée: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ScanSession]:
        with self._lock:
            return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                session.updated_at = datetime.now().isoformat()
                return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False
    
    def list_sessions(self) -> List[ScanSession]:
        with self._lock:
            return list(self._sessions.values())
    
    def acquire_processing_slot(self) -> bool:
        return self._processing_semaphore.acquire(blocking=False)
    
    def release_processing_slot(self):
        self._processing_semaphore.release()

# Gestionnaire de sessions global
session_manager = SessionManager()

class Scanner3D:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"])
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """Calculer le checksum MD5 d'un fichier"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def validate_image(self, file_path: str) -> bool:
        """Valider qu'un fichier est une image valide"""
        try:
            img = cv2.imread(file_path)
            return img is not None and img.size > 0
        except Exception:
            return False
    
    async def extract_features_async(self, image_path: str) -> Dict:
        """Extraction asynchrone des caractéristiques"""
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self.executor, self.extract_features, image_path),
                timeout=CONFIG["FEATURE_EXTRACTION_TIMEOUT"]
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout lors de l'extraction des caractéristiques pour {image_path}")
            return {'error': 'Timeout lors de l\'extraction des caractéristiques'}
    
    def extract_features(self, image_path: str) -> Dict:
        """Extraction des caractéristiques d'une image pour la reconstruction 3D"""
        try:
            # Charger l'image
            img = cv2.imread(image_path)
            if img is None:
                raise Exception(f"Impossible de charger l'image {image_path}")
            
            # Redimensionner si trop grande
            height, width = img.shape[:2]
            if width > 1920 or height > 1080:
                scale = min(1920/width, 1080/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Améliorer le contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Détection des points clés avec ORB (plus robuste)
            orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            # Extraire les coordonnées des points clés avec leurs scores
            points = []
            if keypoints:
                points = [{
                    'x': float(kp.pt[0]),
                    'y': float(kp.pt[1]),
                    'response': float(kp.response),
                    'angle': float(kp.angle),
                    'octave': int(kp.octave)
                } for kp in keypoints]
            
            # Détection des contours pour la structure
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculer des métriques avancées
            structure_metrics = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filtrer les petits contours
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        # Moments pour plus d'informations géométriques
                        moments = cv2.moments(contour)
                        if moments['m00'] != 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])
                        else:
                            cx = cy = 0
                        
                        # Approximation polygonale
                        epsilon = 0.02 * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        structure_metrics.append({
                            'area': float(area),
                            'perimeter': float(perimeter),
                            'circularity': float(4 * np.pi * area / (perimeter * perimeter)),
                            'centroid': [float(cx), float(cy)],
                            'vertices': len(approx),
                            'solidity': float(area / cv2.contourArea(cv2.convexHull(contour)))
                        })
            
            # Calcul d'histogramme pour la texture
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            texture_features = {
                'mean': float(np.mean(gray)),
                'std': float(np.std(gray)),
                'histogram': hist.flatten().tolist()
            }
            
            return {
                'keypoints': points,
                'num_features': len(points),
                'structure_metrics': structure_metrics,
                'texture_features': texture_features,
                'image_size': img.shape[:2],
                'quality_score': len(points) / 2000.0  # Score de qualité basé sur les features
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des caractéristiques: {str(e)}")
            return {'error': str(e)}
    
    async def generate_point_cloud_async(self, features_list: List[Dict], quality: QualityEnum) -> Dict:
        """Génération asynchrone du nuage de points"""
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self.executor, self.generate_point_cloud, features_list, quality),
                timeout=CONFIG["POINT_CLOUD_GENERATION_TIMEOUT"]
            )
        except asyncio.TimeoutError:
            logger.error("Timeout lors de la génération du nuage de points")
            return {'error': 'Timeout lors de la génération du nuage de points'}
    
    def generate_point_cloud(self, features_list: List[Dict], quality: QualityEnum) -> Dict:
        """Génération d'un nuage de points 3D à partir des caractéristiques"""
        try:
            all_points = []
            valid_features = [f for f in features_list if 'error' not in f]
            
            if not valid_features:
                return {'error': 'Aucune caractéristique valide trouvée'}
            
            max_points = {'low': 100, 'medium': 150, 'high': 200}[quality]
            base_radius = {'low': 80, 'medium': 100, 'high': 120}[quality]
            height_variation = {'low': 40, 'medium': 50, 'high': 60}[quality]

            # Paramètres pour la reconstruction 3D
            base_radius = 100
            height_variation = 50
            
            for i, features in enumerate(valid_features):
                # Calculer la position de la caméra dans un cercle
                angle = (i * 2 * np.pi) / len(valid_features)
                camera_x = base_radius * np.cos(angle)
                camera_y = base_radius * np.sin(angle)
                camera_z = height_variation * np.sin(angle * 2)
                
                # Sélectionner les meilleurs points (basé sur le score de réponse)
                keypoints = features['keypoints']
                if len(keypoints) > 150:  # Limiter le nombre de points
                    keypoints = sorted(keypoints, key=lambda k: k['response'], reverse=True)[:150]
                
                for point in keypoints:
                    # Projeter le point 2D en 3D
                    x_2d = point['x'] - features['image_size'][1] / 2
                    y_2d = point['y'] - features['image_size'][0] / 2
                    
                    # Estimation de la profondeur basée sur la réponse du point
                    depth_factor = 1.0 + (point['response'] / 50.0)
                    depth = 80 * depth_factor + np.random.normal(0, 5)
                    
                    # Transformation en coordonnées 3D
                    x_3d = x_2d * depth / 200 + camera_x * 0.1
                    y_3d = y_2d * depth / 200 + camera_y * 0.1
                    z_3d = depth + camera_z * 0.2
                    
                    # Couleur basée sur la profondeur et l'angle
                    color_intensity = int(128 + 127 * np.sin(angle))
                    color = [
                        min(255, max(0, color_intensity)),
                        min(255, max(0, int(128 + depth / 2))),
                        min(255, max(0, int(200 - depth / 2)))
                    ]
                    
                    all_points.append({
                        'x': float(x_3d),
                        'y': float(y_3d),
                        'z': float(z_3d),
                        'color': color,
                        'confidence': float(point['response'] / 50.0)
                    })
            
            # Nettoyer les points aberrants
            if all_points:
                all_points = self.remove_outliers(all_points)
            
            return {
                'points': all_points,
                'num_points': len(all_points),
                'bounds': self.calculate_bounds(all_points) if all_points else None,
                'quality_metrics': self.calculate_quality_metrics(all_points, valid_features)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du nuage de points: {str(e)}")
            return {'error': str(e)}
    
    def remove_outliers(self, points: List[Dict], threshold: float = 2.0) -> List[Dict]:
        """Supprimer les points aberrants du nuage de points"""
        if len(points) < 10:
            return points
        
        # Calculer les statistiques
        coords = np.array([[p['x'], p['y'], p['z']] for p in points])
        mean = np.mean(coords, axis=0)
        std = np.std(coords, axis=0)
        
        # Filtrer les points qui sont dans les limites acceptables
        filtered_points = []
        for i, point in enumerate(points):
            coord = coords[i]
            if all(abs(coord[j] - mean[j]) <= threshold * std[j] for j in range(3)):
                filtered_points.append(point)
        
        logger.info(f"Filtrage des outliers: {len(points)} -> {len(filtered_points)} points")
        return filtered_points
    
    def calculate_bounds(self, points: List[Dict]) -> Dict:
        """Calculer les limites du nuage de points"""
        if not points:
            return None
        
        x_coords = [p['x'] for p in points]
        y_coords = [p['y'] for p in points]
        z_coords = [p['z'] for p in points]
        
        return {
            'min_x': float(min(x_coords)),
            'max_x': float(max(x_coords)),
            'min_y': float(min(y_coords)),
            'max_y': float(max(y_coords)),
            'min_z': float(min(z_coords)),
            'max_z': float(max(z_coords)),
            'center': [
                float(np.mean(x_coords)),
                float(np.mean(y_coords)),
                float(np.mean(z_coords))
            ]
        }
    
    def calculate_quality_metrics(self, points: List[Dict], features_list: List[Dict]) -> Dict:
        """Calculer les métriques de qualité du modèle"""
        if not points or not features_list:
            return {}
        
        # Densité des points
        bounds = self.calculate_bounds(points)
        if bounds:
            volume = ((bounds['max_x'] - bounds['min_x']) * 
                     (bounds['max_y'] - bounds['min_y']) * 
                     (bounds['max_z'] - bounds['min_z']))
            density = len(points) / max(volume, 1)
        else:
            density = 0
        
        # Score de qualité moyen des images
        quality_scores = [f.get('quality_score', 0) for f in features_list]
        avg_quality = np.mean(quality_scores)
        
        # Couverture (basée sur le nombre d'images)
        coverage_score = min(len(features_list) / 20.0, 1.0)  # Optimal à 20 images
        
        return {
            'point_density': float(density),
            'average_image_quality': float(avg_quality),
            'coverage_score': float(coverage_score),
            'overall_quality': float((density/1000 + avg_quality + coverage_score) / 3)
        }

# Instance du scanner
scanner = Scanner3D()

# Gestionnaire de contexte pour l'initialisation et le nettoyage
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage
    logger.info("Démarrage du serveur Scanner 3D")
    # Nettoyer les fichiers temporaires au démarrage
    await cleanup_temp_files()
    # Programmer le nettoyage périodique
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Arrêt
    logger.info("Arrêt du serveur Scanner 3D")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

# Application FastAPI
app = FastAPI(
    title="Scanner 3D API",
    description="API pour la numérisation 3D à partir d'images",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fonction de nettoyage
async def cleanup_temp_files():
    """Nettoyer les fichiers temporaires"""
    try:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            TEMP_DIR.mkdir()
        logger.info("Fichiers temporaires nettoyés")
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")

async def periodic_cleanup():
    """Nettoyage périodique des sessions expirées"""
    while True:
        try:
            await asyncio.sleep(CONFIG["CLEANUP_INTERVAL"])
            current_time = datetime.now()
            
            expired_sessions = []
            for session in session_manager.list_sessions():
                session_time = datetime.fromisoformat(session.created_at)
                if (current_time - session_time).total_seconds() > CONFIG["SESSION_TIMEOUT"]:
                    expired_sessions.append(session.id)
            
            for session_id in expired_sessions:
                await cleanup_session(session_id)
                logger.info(f"Session expirée nettoyée: {session_id}")
                
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage périodique: {e}")

async def cleanup_session(session_id: str):
    """Nettoyer les fichiers d'une session"""
    session = session_manager.get_session(session_id)
    if session:
        # Supprimer les fichiers d'images
        for image in session.images:
            try:
                if os.path.exists(image.path):
                    os.remove(image.path)
            except Exception as e:
                logger.error(f"Erreur suppression image {image.path}: {e}")
        
        # Supprimer le modèle
        if session.model_path and os.path.exists(session.model_path):
            try:
                os.remove(session.model_path)
            except Exception as e:
                logger.error(f"Erreur suppression modèle {session.model_path}: {e}")
        
        # Supprimer de la mémoire
        session_manager.delete_session(session_id)


# Routes API

@app.get("/")
async def root():
    """Point d'entrée principal"""
    return {
        "message": "Scanner 3D API - Serveur actif",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "start_scan": "/api/scan/start",
            "upload": "/api/scan/{session_id}/upload",
            "process": "/api/scan/{session_id}/process",
            "status": "/api/scan/{session_id}/status",
            "model": "/api/scan/{session_id}/model",
            "thumbnail": "/api/scan/{session_id}/thumbnail",
            "scans": "/api/scans"
        }
    }

@app.get("/api/health")
async def health_check():
    """Vérification de l'état du serveur"""
    sessions = session_manager.list_sessions()

    if not sessions:
        name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        session_id = session_manager.create_session(name=name)
        sessions = session_manager.list_sessions()

    sorted_sessions = sorted(sessions, key=lambda s: s.created_at, reverse=True)
    current = sorted_sessions[0]

    current_session = {
        "session_id": current.id,
        "status": current.status.value,
        "created_at": current.created_at,
        "photos_count": len(current.images),
        "name": current.name
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions),
        "current_session": current_session,  # Nouvelle information
        "config": {
            "max_upload_size": CONFIG["MAX_UPLOAD_SIZE"],
            "max_images_per_session": CONFIG["MAX_IMAGES_PER_SESSION"],
            "supported_formats": CONFIG["SUPPORTED_FORMATS"]
        }
    }

@app.post("/api/scan/start")
async def start_scan(request: StartScanRequest):
    """Démarrer une nouvelle session de scan"""
    try:
        session_id = session_manager.create_session(request.name)
        session = session_manager.get_session(session_id)
        
        return {
            "session_id": session_id,
            "message": "Session de scan démarrée",
            "name": session.name,
            "status": session.status.value,
            "created_at": session.created_at
        }
    except Exception as e:
        logger.error(f"Erreur création session: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création de la session: {str(e)}")

@app.post("/api/scan/{session_id}/upload")
async def upload_scan_image(session_id: str, file: UploadFile = File(...)):
    """Uploader une image pour le scan"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session de scan non trouvée")
    
    if session.status != ScanStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Session de scan non active")
    
    # Vérifier les limites
    if len(session.images) >= CONFIG["MAX_IMAGES_PER_SESSION"]:
        raise HTTPException(status_code=400, detail="Nombre maximum d'images atteint")
    
    # Vérifier l'extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in CONFIG["SUPPORTED_FORMATS"]:
        raise HTTPException(status_code=400, detail=f"Format non supporté. Formats acceptés: {CONFIG['SUPPORTED_FORMATS']}")
    
    try:
        # Lire le contenu du fichier
        content = await file.read()
        
        # Vérifier la taille
        if len(content) > CONFIG["MAX_UPLOAD_SIZE"]:
            raise HTTPException(status_code=400, detail="Fichier trop volumineux")
        
        # Générer un nom de fichier unique
        image_filename = f"{session_id}_{uuid.uuid4()}{file_ext}"
        image_path = UPLOAD_DIR / image_filename
        
        # Sauvegarder l'image
        async with aiofiles.open(image_path, 'wb') as buffer:
            await buffer.write(content)
        
        # Valider l'image
        if not scanner.validate_image(str(image_path)):
            os.remove(image_path)
            raise HTTPException(status_code=400, detail="Fichier image invalide")
        
        # Calculer le checksum
        checksum = scanner.calculate_file_checksum(str(image_path))
        
        # Créer l'objet image
        scan_image = ScanImage(
            filename=image_filename,
            path=str(image_path),
            uploaded_at=datetime.now().isoformat(),
            file_size=len(content),
            checksum=checksum
        )
        
        # Ajouter à la session
        session.images.append(scan_image)
        
        # Mettre à jour le progrès
        progress = min(len(session.images) * 5, 90)  # 5% par image, max 90%
        session_manager.update_session(session_id, progress=progress)
        
        logger.info(f"Image uploadée pour la session {session_id}: {image_filename}")
        
        return {
            "message": "Image uploadée avec succès",
            "filename": image_filename,
            "file_size": len(content),
            "checksum": checksum,
            "photos_count": len(session.images),
            "progress": progress
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'upload: {str(e)}")

@app.post("/api/scan/{session_id}/process")
async def process_scan(session_id: str, request: ProcessScanRequest, background_tasks: BackgroundTasks):
    """Traiter les images pour créer le modèle 3D"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session de scan non trouvée")
    
    if not session.images:
        raise HTTPException(status_code=400, detail="Aucune image à traiter")
    
    if session.status == ScanStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Traitement déjà en cours")
    
    # Vérifier la disponibilité des slots de traitement
    if not session_manager.acquire_processing_slot():
        raise HTTPException(status_code=503, detail="Trop de traitements en cours, réessayez plus tard")
    
    try:
        # Mettre à jour le statut
        session_manager.update_session(
            session_id,
            status=ScanStatus.PROCESSING,
            processing_start_time=datetime.now().isoformat(),
            progress=0
        )
        
        # Lancer le traitement en arrière-plan
        background_tasks.add_task(process_scan_background, session_id, request.quality)
        
        return {
            "message": "Traitement démarré",
            "session_id": session_id,
            "status": "processing",
            "image_count": len(session.images)
        }
    
    except Exception as e:
        session_manager.release_processing_slot()
        logger.error(f"Erreur lors du démarrage du traitement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du démarrage du traitement: {str(e)}")


async def process_scan_background(session_id: str, quality: str):
    """Traitement en arrière-plan du scan"""
    start_time = datetime.now()
    
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return
        
        logger.info(f"Début du traitement pour la session {session_id}")
        
        # Extraire les caractéristiques de chaque image
        features_list = []
        for i, image_info in enumerate(session.images):
            try:
                features = await scanner.extract_features_async(image_info.path)
                features_list.append(features)
                
                # Mettre à jour le progrès (0-70% pour l'extraction)
                progress = (i + 1) * 70 / len(session.images)
                session_manager.update_session(session_id, progress=progress)
                
            except Exception as e:
                logger.error(f"Erreur extraction features pour {image_info.filename}: {e}")
                features_list.append({'error': str(e)})
        
        # Générer le nuage de points 3D
        logger.info(f"Génération du nuage de points pour la session {session_id}")
        session_manager.update_session(session_id, progress=75)
        
        point_cloud = await scanner.generate_point_cloud_async(features_list)
        if 'error' in point_cloud:
            raise Exception(point_cloud['error'])
        
        # Sauvegarder le modèle
        model_filename = f"{session_id}_model_{quality}.ply"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        # Créer le dossier des modèles s'il n'existe pas
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        session_manager.update_session(session_id, progress=85)
        
        # Sauvegarder le nuage de points au format PLY
        await save_point_cloud_ply(point_cloud, model_path)
        
        # Générer les métadonnées du modèle
        model_metadata = {
            "filename": model_filename,
            "path": model_path,
            "quality": quality,
            "point_count": len(point_cloud.get('points', [])),
            "created_at": datetime.now().isoformat(),
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "bounding_box": calculate_bounding_box(point_cloud),
            "file_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0
        }
        
        session_manager.update_session(session_id, progress=95)
        
        # Générer une miniature du modèle
        thumbnail_path = await generate_model_thumbnail(model_path, session_id)
        if thumbnail_path:
            model_metadata["thumbnail_path"] = thumbnail_path
        
        # Finaliser la session
        session_manager.update_session(
            session_id,
            status=ScanStatus.COMPLETED,
            progress=100,
            model_path=model_path,
            model_metadata=model_metadata,
            processing_end_time=datetime.now().isoformat()
        )
        
        logger.info(f"Traitement terminé pour la session {session_id} en {model_metadata['processing_time']:.2f}s")
        
    except Exception as e:
        error_message = f"Erreur lors du traitement: {str(e)}"
        logger.error(f"Session {session_id}: {error_message}")
        
        # Mettre à jour le statut d'erreur
        session_manager.update_session(
            session_id,
            status=ScanStatus.ERROR,
            error_message=error_message,
            processing_end_time=datetime.now().isoformat()
        )
        
    finally:
        # Libérer le slot de traitement
        session_manager.release_processing_slot()


# async def save_point_cloud_ply(point_cloud: dict, output_path: str):
#     """Sauvegarder le nuage de points au format PLY"""
#     try:
#         points = point_cloud.get('points', [])
#         colors = point_cloud.get('colors', [])
        
#         with open(output_path, 'w') as f:
#             # En-tête PLY
#             f.write("ply\n")
#             f.write("format ascii 1.0\n")
#             f.write(f"element vertex {len(points)}\n")
#             f.write("property float x\n")
#             f.write("property float y\n")
#             f.write("property float z\n")
            
#             if colors:
#                 f.write("property uchar red\n")
#                 f.write("property uchar green\n")
#                 f.write("property uchar blue\n")
            
#             f.write("end_header\n")
            
#             # Données des points
#             for i, point in enumerate(points):
#                 line = f"{point[0]} {point[1]} {point[2]}"
#                 if colors and i < len(colors):
#                     color = colors[i]
#                     line += f" {color[0]} {color[1]} {color[2]}"
#                 f.write(line + "\n")
                
#     except Exception as e:
#         raise Exception(f"Erreur lors de la sauvegarde PLY: {str(e)}")

async def save_point_cloud_ply(point_cloud: dict, output_path: str):
    try:
        points = point_cloud.get('points', [])
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point in points:
                f.write(f"{point['x']} {point['y']} {point['z']} {point['color'][0]} {point['color'][1]} {point['color'][2]}\n")
    except Exception as e:
        raise Exception(f"Erreur lors de la sauvegarde PLY: {str(e)}")


def calculate_bounding_box(point_cloud: dict) -> dict:
    """Calculer la boîte englobante du nuage de points"""
    try:
        points = point_cloud.get('points', [])
        if not points:
            return {}
        
        points_array = np.array(points)
        min_coords = np.min(points_array, axis=0)
        max_coords = np.max(points_array, axis=0)
        
        return {
            "min": min_coords.tolist(),
            "max": max_coords.tolist(),
            "center": ((min_coords + max_coords) / 2).tolist(),
            "dimensions": (max_coords - min_coords).tolist()
        }
        
    except Exception as e:
        logger.error(f"Erreur calcul bounding box: {str(e)}")
        return {}


async def generate_model_thumbnail(model_path: str, session_id: str) -> str:
    """Générer une miniature du modèle 3D"""
    try:
        thumbnail_filename = f"{session_id}_thumbnail.png"
        thumbnail_path = os.path.join(THUMBNAILS_DIR, thumbnail_filename)
        
        # Créer le dossier des miniatures s'il n'existe pas
        os.makedirs(THUMBNAILS_DIR, exist_ok=True)
        
        # Utiliser Open3D pour générer la miniature
        import open3d as o3d
        
        # Charger le modèle PLY
        pcd = o3d.io.read_point_cloud(model_path)
        
        if len(pcd.points) == 0:
            logger.warning(f"Modèle vide pour la session {session_id}")
            return None
        
        # Créer un visualiseur off-screen
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=512, height=512)
        vis.add_geometry(pcd)
        
        # Ajuster la caméra
        vis.get_view_control().set_front([0, 0, -1])
        vis.get_view_control().set_up([0, -1, 0])
        vis.get_view_control().set_lookat([0, 0, 0])
        vis.get_view_control().set_zoom(0.8)
        
        # Capturer l'image
        vis.capture_screen_image(thumbnail_path)
        vis.destroy_window()
        
        return thumbnail_path
        
    except Exception as e:
        logger.error(f"Erreur génération miniature: {str(e)}")
        return None

# Ajouter un décorateur pour la gestion d'erreurs
@contextmanager
async def cleanup_on_error(session_id: str, file_path: str = None):
    try:
        yield
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        session_manager.update_session(session_id, status=ScanStatus.ERROR, error_message=str(e))
        raise

# Validation d'image plus robuste
def validate_image_advanced(image_path: str) -> dict:
    """Validation avancée avec métadonnées"""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return {
                "valid": True,
                "format": img.format,
                "size": img.size,
                "mode": img.mode,
                "dpi": img.info.get('dpi', (72, 72))
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}

# Streaming pour les gros fichiers
async def stream_upload(file: UploadFile, destination: str, chunk_size: int = 8192):
    """Upload par chunks pour éviter la surcharge mémoire"""
    async with aiofiles.open(destination, 'wb') as buffer:
        while chunk := await file.read(chunk_size):
            await buffer.write(chunk)

# Traitement parallèle des features
async def extract_features_parallel(image_paths: List[str], max_workers: int = 4):
    """Extraction parallèle des caractéristiques"""
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_image(path):
        async with semaphore:
            return await scanner.extract_features_async(path)
    
    return await asyncio.gather(*[process_image(path) for path in image_paths])

# Validation sécurisée des fichiers
def secure_filename(filename: str) -> str:
    """Sécuriser le nom de fichier"""
    import re
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return filename[:100]  # Limiter la longueur

# Validation du type MIME
def validate_mime_type(file_path: str) -> bool:
    """Valider le type MIME réel du fichier"""
    import magic
    mime_type = magic.from_file(file_path, mime=True)
    return mime_type in ['image/jpeg', 'image/png', 'image/bmp']


limiter = Limiter(key_func=get_remote_address)

@app.get("/api/scan/{session_id}/status")
async def get_scan_status(session_id: str):
    """Obtenir le statut du traitement"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session non trouvée")

    response = {
        "session_id": session.id,
        "status": session.status.value,
        "progress": session.progress,
        "image_count": len(session.images),
        "created_at": session.created_at
    }

    if session.processing_start_time is not None:
        response["processing_start_time"] = session.processing_start_time

    if session.processing_end_time is not None:
        response["processing_end_time"] = session.processing_end_time

    if session.total_processing_time is not None:
        response["total_processing_time"] = session.total_processing_time

    if session.error_message is not None and session.status == ScanStatus.ERROR:
        response["error_message"] = session.error_message

    if session.status == ScanStatus.COMPLETED and session.model_path and session.model_filename:
        response["model_metadata"] = {
            "path": session.model_path,
            "filename": session.model_filename
        }

    return response



@app.get("/api/scan/{session_id}/download")
async def download_model(session_id: str):
    """Télécharger le modèle 3D généré"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    
    if session.status != ScanStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Modèle non disponible")
    
    if not session.model_path or not os.path.exists(session.model_path):
        raise HTTPException(status_code=404, detail="Fichier modèle non trouvé")
    
    return FileResponse(
        session.model_path,
        media_type="application/octet-stream",
        filename=os.path.basename(session.model_path)
    )


@app.get("/api/scan/{session_id}/thumbnail")
async def get_thumbnail(session_id: str):
    """Obtenir la miniature du modèle"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    
    if (session.status != ScanStatus.COMPLETED or 
        not session.model_metadata or 
        'thumbnail_path' not in session.model_metadata):
        raise HTTPException(status_code=404, detail="Miniature non disponible")
    
    thumbnail_path = session.model_metadata['thumbnail_path']
    if not os.path.exists(thumbnail_path):
        raise HTTPException(status_code=404, detail="Fichier miniature non trouvé")
    
    return FileResponse(thumbnail_path, media_type="image/png")

# Endpoint pour obtenir les estimations
@app.get("/api/scan/{session_id}/estimate")
async def get_processing_estimate(session_id: str):
    """Estimer le temps de traitement"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    
    # Estimation basée sur le nombre d'images et les performances historiques
    base_time_per_image = 30  # secondes
    estimated_time = len(session.images) * base_time_per_image
    
    return {
        "estimated_processing_time": estimated_time,
        "queue_position": session_manager.get_queue_position(session_id),
        "average_wait_time": session_manager.get_average_wait_time()
    }

# WebSocket pour le statut en temps réel
@app.websocket("/ws/scan/{session_id}")
async def websocket_scan_status(websocket: WebSocket, session_id: str):
    """WebSocket pour le suivi en temps réel"""
    await websocket.accept()
    
    while True:
        session = session_manager.get_session(session_id)
        if not session:
            break
            
        await websocket.send_json({
            "status": session.status.value,
            "progress": session.progress,
            "message": get_status_message(session.status, session.progress)
        })
        
        if session.status in [ScanStatus.COMPLETED, ScanStatus.ERROR]:
            break
            
        await asyncio.sleep(1)

@dataclass
class CameraPose:
    """Pose de la caméra (rotation + translation)"""
    rotation: np.ndarray  # Matrice 3x3
    translation: np.ndarray  # Vecteur 3x1
    
    def to_projection_matrix(self, intrinsics: np.ndarray) -> np.ndarray:
        """Créer la matrice de projection"""
        extrinsics = np.hstack([self.rotation, self.translation.reshape(-1, 1)])
        return intrinsics @ extrinsics

class ImprovedScanner3D:
    """Scanner 3D avec reconstruction améliorée basée sur Structure from Motion"""
    
    def __init__(self):
        self.min_match_count = 10
        self.max_reproj_error = 1.0
        self.min_triangulation_angle = 5.0  # degrés
        
        # Configurateur de détecteur de features
        self.detector = cv2.SIFT_create(
            nfeatures=5000,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        
        # Matcher pour correspondances
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
    def estimate_camera_intrinsics(self, image_size: Tuple[int, int]) -> CameraParameters:
        """Estimer les paramètres intrinsèques basés sur la taille d'image"""
        width, height = image_size
        
        # Estimation basée sur des valeurs typiques
        focal_length = max(width, height) * 1.2
        
        return CameraParameters(
            fx=focal_length,
            fy=focal_length,
            cx=width / 2,
            cy=height / 2
        )
    
    def extract_features_sift(self, image_path: str) -> Dict:
        """Extraction de features avec SIFT"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise Exception(f"Impossible de charger {image_path}")
            
            # Préprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Amélioration du contraste
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Détection et description
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            
            if descriptors is None:
                return {'error': 'Aucune feature détectée'}
            
            # Convertir en format sérialisable
            kp_data = []
            for kp in keypoints:
                kp_data.append({
                    'x': float(kp.pt[0]),
                    'y': float(kp.pt[1]),
                    'response': float(kp.response),
                    'angle': float(kp.angle),
                    'octave': int(kp.octave),
                    'size': float(kp.size)
                })
            
            return {
                'keypoints': kp_data,
                'descriptors': descriptors.tolist(),
                'image_size': img.shape[:2],
                'num_features': len(keypoints)
            }
            
        except Exception as e:
            logger.error(f"Erreur extraction SIFT: {str(e)}")
            return {'error': str(e)}
    
    def match_features(self, features1: Dict, features2: Dict) -> List[Tuple[int, int]]:
        """Matcher les features entre deux images"""
        if 'error' in features1 or 'error' in features2:
            return []
        
        desc1 = np.array(features1['descriptors'], dtype=np.float32)
        desc2 = np.array(features2['descriptors'], dtype=np.float32)
        
        if len(desc1) < 10 or len(desc2) < 10:
            return []
        
        try:
            # Matching avec ratio test
            matches = self.matcher.match(desc1, desc2)
            
            # Filtrer les bonnes correspondances
            good_matches = []
            for match in matches:
                if match.distance < 0.7:  # Seuil de distance
                    good_matches.append((match.queryIdx, match.trainIdx))
            
            return good_matches
            
        except Exception as e:
            logger.error(f"Erreur matching: {str(e)}")
            return []
    
    def estimate_fundamental_matrix(self, points1: np.ndarray, points2: np.ndarray) -> Optional[np.ndarray]:
        """Estimer la matrice fondamentale avec RANSAC"""
        if len(points1) < 8 or len(points2) < 8:
            return None
        
        try:
            F, mask = cv2.findFundamentalMat(
                points1, points2,
                cv2.FM_RANSAC,
                ransacReprojThreshold=1.0,
                confidence=0.99
            )
            
            if F is None or F.shape != (3, 3):
                return None
            
            return F
            
        except Exception as e:
            logger.error(f"Erreur estimation matrice fondamentale: {str(e)}")
            return None
    
    def estimate_poses_from_fundamental(self, F: np.ndarray, points1: np.ndarray, 
                                      points2: np.ndarray, intrinsics: np.ndarray) -> List[CameraPose]:
        """Estimer les poses des caméras à partir de la matrice fondamentale"""
        try:
            # Calculer la matrice essentielle
            E = intrinsics.T @ F @ intrinsics
            
            # Décomposer la matrice essentielle
            _, R1, R2, t = cv2.recoverPose(E, points1, points2, intrinsics)
            
            # Tester les 4 configurations possibles
            possible_poses = [
                CameraPose(R1, t),
                CameraPose(R1, -t),
                CameraPose(R2, t),
                CameraPose(R2, -t)
            ]
            
            # Garder seulement les poses valides (points devant les caméras)
            valid_poses = []
            for pose in possible_poses:
                if self.validate_pose(pose, points1, points2, intrinsics):
                    valid_poses.append(pose)
            
            return valid_poses
            
        except Exception as e:
            logger.error(f"Erreur estimation poses: {str(e)}")
            return []
    
    def validate_pose(self, pose: CameraPose, points1: np.ndarray, 
                     points2: np.ndarray, intrinsics: np.ndarray) -> bool:
        """Valider une pose en vérifiant que les points sont devant les caméras"""
        try:
            # Projection matrices
            P1 = intrinsics @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = pose.to_projection_matrix(intrinsics)
            
            # Trianguler quelques points
            sample_size = min(20, len(points1))
            indices = np.random.choice(len(points1), sample_size, replace=False)
            
            points_3d = cv2.triangulatePoints(
                P1, P2, 
                points1[indices].T, 
                points2[indices].T
            )
            
            # Convertir en coordonnées cartésiennes
            points_3d = points_3d[:3] / points_3d[3]
            
            # Vérifier que les points sont devant les deux caméras
            # Camera 1 (origine)
            depths1 = points_3d[2]
            
            # Camera 2
            points_cam2 = pose.rotation @ points_3d + pose.translation.reshape(-1, 1)
            depths2 = points_cam2[2]
            
            # Au moins 80% des points doivent être devant les deux caméras
            valid_count = np.sum((depths1 > 0) & (depths2 > 0))
            return valid_count / len(depths1) > 0.8
            
        except Exception as e:
            logger.error(f"Erreur validation pose: {str(e)}")
            return False
    
    def triangulate_points(self, matches: List[Tuple[int, int]], 
                          features1: Dict, features2: Dict, 
                          pose1: CameraPose, pose2: CameraPose, 
                          intrinsics: np.ndarray) -> np.ndarray:
        """Trianguler les points 3D à partir des correspondances"""
        if not matches:
            return np.array([])
        
        # Extraire les points correspondants
        points1 = []
        points2 = []
        
        for idx1, idx2 in matches:
            kp1 = features1['keypoints'][idx1]
            kp2 = features2['keypoints'][idx2]
            points1.append([kp1['x'], kp1['y']])
            points2.append([kp2['x'], kp2['y']])
        
        points1 = np.array(points1, dtype=np.float32)
        points2 = np.array(points2, dtype=np.float32)
        
        try:
            # Matrices de projection
            P1 = pose1.to_projection_matrix(intrinsics)
            P2 = pose2.to_projection_matrix(intrinsics)
            
            # Triangulation
            points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
            
            # Convertir en coordonnées 3D
            points_3d = points_4d[:3] / points_4d[3]
            
            # Filtrer les points aberrants
            points_3d = self.filter_outlier_points(points_3d.T)
            
            return points_3d
            
        except Exception as e:
            logger.error(f"Erreur triangulation: {str(e)}")
            return np.array([])
    
    def filter_outlier_points(self, points_3d: np.ndarray, 
                            max_distance: float = 1000.0) -> np.ndarray:
        """Filtrer les points aberrants du nuage de points"""
        if len(points_3d) == 0:
            return points_3d
        
        # Filtrer par distance maximale depuis l'origine
        distances = np.linalg.norm(points_3d, axis=1)
        mask = distances < max_distance
        points_3d = points_3d[mask]
        
        if len(points_3d) < 10:
            return points_3d
        
        # Clustering pour enlever les points isolés
        try:
            clustering = DBSCAN(eps=50, min_samples=5)
            labels = clustering.fit_predict(points_3d)
            
            # Garder le plus gros cluster
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                main_cluster = unique_labels[np.argmax(counts)]
                points_3d = points_3d[labels == main_cluster]
            
        except Exception as e:
            logger.warning(f"Erreur clustering: {str(e)}")
        
        return points_3d
    
    def reconstruct_3d_model(self, features_list: List[Dict]) -> Dict:
        """Reconstruction 3D complète avec Structure from Motion"""
        try:
            if len(features_list) < 2:
                return {'error': 'Au moins 2 images nécessaires'}
            
            # Filtrer les features valides
            valid_features = [f for f in features_list if 'error' not in f]
            if len(valid_features) < 2:
                return {'error': 'Pas assez d\'images valides'}
            
            # Estimer les paramètres intrinsèques (supposés identiques)
            image_size = valid_features[0]['image_size']
            intrinsics_params = self.estimate_camera_intrinsics(image_size)
            intrinsics = intrinsics_params.to_matrix()
            
            # Reconstruction incrémentale
            all_points_3d = []
            camera_poses = [CameraPose(np.eye(3), np.zeros(3))]  # Première caméra à l'origine
            
            # Traiter chaque paire d'images consécutives
            for i in range(len(valid_features) - 1):
                features1 = valid_features[i]
                features2 = valid_features[i + 1]
                
                # Matcher les features
                matches = self.match_features(features1, features2)
                
                if len(matches) < self.min_match_count:
                    logger.warning(f"Pas assez de correspondances entre images {i} et {i+1}")
                    continue
                
                # Préparer les points pour l'estimation
                points1 = np.array([[features1['keypoints'][m[0]]['x'], 
                                   features1['keypoints'][m[0]]['y']] for m in matches])
                points2 = np.array([[features2['keypoints'][m[1]]['x'], 
                                   features2['keypoints'][m[1]]['y']] for m in matches])
                
                # Estimer la matrice fondamentale
                F = self.estimate_fundamental_matrix(points1, points2)
                if F is None:
                    continue
                
                # Estimer les poses
                poses = self.estimate_poses_from_fundamental(F, points1, points2, intrinsics)
                if not poses:
                    continue
                
                # Prendre la première pose valide
                pose2 = poses[0]
                camera_poses.append(pose2)
                
                # Trianguler les points
                points_3d = self.triangulate_points(matches, features1, features2, 
                                                  camera_poses[i], pose2, intrinsics)
                
                if len(points_3d) > 0:
                    all_points_3d.extend(points_3d)
            
            # Fusionner et nettoyer tous les points
            if all_points_3d:
                all_points_3d = np.array(all_points_3d)
                all_points_3d = self.filter_outlier_points(all_points_3d)
                
                # Générer les couleurs
                colors = self.generate_point_colors(all_points_3d)
                
                # Convertir en format de sortie
                points_output = []
                for i, point in enumerate(all_points_3d):
                    color = colors[i] if i < len(colors) else [128, 128, 128]
                    points_output.append({
                        'x': float(point[0]),
                        'y': float(point[1]),
                        'z': float(point[2]),
                        'color': color,
                        'confidence': 1.0
                    })
                
                return {
                    'points': points_output,
                    'num_points': len(points_output),
                    'camera_poses': len(camera_poses),
                    'reconstruction_method': 'structure_from_motion'
                }
            
            return {'error': 'Aucun point 3D reconstruit'}
            
        except Exception as e:
            logger.error(f"Erreur reconstruction 3D: {str(e)}")
            return {'error': str(e)}
    
    def generate_point_colors(self, points_3d: np.ndarray) -> List[List[int]]:
        """Générer des couleurs pour les points basées sur leur position"""
        if len(points_3d) == 0:
            return []
        
        # Normaliser les coordonnées
        min_coords = np.min(points_3d, axis=0)
        max_coords = np.max(points_3d, axis=0)
        normalized = (points_3d - min_coords) / (max_coords - min_coords + 1e-8)
        
        colors = []
        for point in normalized:
            # Mapping position -> couleur
            r = int(255 * point[0])
            g = int(255 * point[1])
            b = int(255 * point[2])
            
            # Assurer que les valeurs sont dans [0, 255]
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            
            colors.append([r, g, b])
        
        return colors