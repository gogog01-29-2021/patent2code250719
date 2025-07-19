from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import secrets
import string

class AuthService:
    def __init__(self, app):
        self.jwt = JWTManager(app)
        app.config['JWT_SECRET_KEY'] = secrets.token_urlsafe(32)
        app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
    
    def register_user(self, username: str, email: str, password: str) -> Dict:
        """사용자 등록"""
        # 중복 확인
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            return {'success': False, 'error': 'User already exists'}
        
        # 비밀번호 해시화
        password_hash = generate_password_hash(password)
        
        # 사용자 생성
        user = User(
            username=username,
            email=email,
            password_hash=password_hash
        )
        
        db.session.add(user)
        db.session.commit()
        
        # 액세스 토큰 생성
        access_token = create_access_token(identity=user.id)
        
        return {
            'success': True,
            'user_id': user.id,
            'access_token': access_token
        }
    
    def login_user(self, email: str, password: str) -> Dict:
        """사용자 로그인"""
        user = User.query.filter_by(email=email).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return {'success': False, 'error': 'Invalid credentials'}
        
        if not user.is_active:
            return {'success': False, 'error': 'Account is deactivated'}
        
        # 로그인 정보 업데이트
        user.last_login_at = datetime.utcnow()
        user.login_count += 1
        db.session.commit()
        
        # 액세스 토큰 생성
        access_token = create_access_token(identity=user.id)
        
        return {
            'success': True,
            'user_id': user.id,
            'access_token': access_token,
            'user_info': {
                'username': user.username,
                'email': user.email,
                'role': user.role
            }
        }
    
    def generate_api_key(self, user_id: int, key_name: str, permissions: Dict) -> Dict:
        """API 키 생성"""
        # API 키 생성
        key = 'pk_' + ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        key_hash = generate_password_hash(key)
        key_prefix = key[:8]
        
        api_key = APIKey(
            user_id=user_id,
            key_name=key_name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            permissions=permissions
        )
        
        db.session.add(api_key)
        db.session.commit()
        
        return {
            'success': True,
            'api_key': key,
            'key_id': api_key.id,
            'key_prefix': key_prefix
        }