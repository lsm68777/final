#!/usr/bin/env python3
# -*- coding: utf-8 -*-   
"""
🚀 프로젝트 피닉스 - 완성된 웹훅 서버 (Phoenix 95점 엔진 완전 연동)
실제 바이낸스 데이터 + Phoenix 95점 AI 분석 + 고급 신호 처리 + 켈리 공식
최고 수준의 실시간 신호 수신 및 거래 실행 시스템 - 완전 안정화 버전
"""

import asyncio
import json
import time
import threading
import aiohttp
import requests
import numpy as np
import signal
import sys
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from collections import deque
import logging
import hashlib
import hmac
from pathlib import Path
import traceback
import gc

# FastAPI 임포트
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel, field_validator

# Phoenix 95점 AI 엔진 임포트
try:
    from phoenix_95_engine import Phoenix95Engine
    PHOENIX_95_AVAILABLE = True
    print("✅ Phoenix 95점 엔진 로드 성공")
except ImportError as e:
    PHOENIX_95_AVAILABLE = False
    print(f"❌ Phoenix 95점 엔진 로드 실패: {e}")
    print("⚠️ 기본 분석 모드로 동작")

# ngrok 연동 설정
NGROK_CONFIG = {
    "enabled": True,
    "auth_token": "2ve8GVHeZA0PtmaCbnS0RIOjwQT_61k2QrkuebZn1ACpz8uum",  # ngrok 계정 토큰
    "region": "ap",    # 아시아-태평양
    "tunnel_name": "phoenix95-webhook",
    "domain": "signor.ngrok.io"  # 유료 도메인
}

# ngrok 터널 시작 함수
async def start_ngrok_tunnel(port: int):
    """ngrok 터널 시작"""
    try:
        import pyngrok
        from pyngrok import ngrok
        
        if NGROK_CONFIG["auth_token"]:
            ngrok.set_auth_token(NGROK_CONFIG["auth_token"])
        
        # 기존 터널 종료
        ngrok.kill()
        
        # 새 터널 시작
        tunnel_options = {
            "region": NGROK_CONFIG["region"],
            "name": NGROK_CONFIG["tunnel_name"]
        }
        
        # 유료 도메인 사용
        if NGROK_CONFIG["domain"]:
            tunnel = ngrok.connect(
                port, 
                "http",
                options={**tunnel_options, "hostname": NGROK_CONFIG["domain"]}
            )
            public_url = f"https://{NGROK_CONFIG['domain']}"
        else:
            tunnel = ngrok.connect(port, "http", options=tunnel_options)
            public_url = tunnel.public_url
        logging.info(f"🌐 ngrok 터널 생성 성공: {public_url}")
        logging.info(f"📡 TradingView 웹훅 URL: {public_url}/webhook/signal")
        
        return public_url
        
    except ImportError:
        logging.warning("⚠️ pyngrok 라이브러리가 설치되지 않음")
        logging.info("💡 설치 명령어: pip install pyngrok")
        return None
    except Exception as e:
        logging.error(f"❌ ngrok 터널 생성 실패: {e}")
        return None

COMPLETE_WEBHOOK_CONFIG = {
    "environment": "complete_phoenix95_production",
    "host": "0.0.0.0",
    "port": 8101,
    "workers": 1,
    "log_level": "info",
    "reload": False,
    "access_log": True,
    "max_connections": 1000,
    "request_timeout": 60,
    "signal_queue_size": 5000,
    "batch_processing": True,
    "phoenix_95_enabled": PHOENIX_95_AVAILABLE,
    "real_data_validation": True,
    "security_enabled": True,
    "rate_limiting": True,
    "health_check_interval": 15,
    "memory_cleanup_interval": 180,
    "auto_restart_on_error": True,
    "graceful_shutdown": True,
    "auto_scaling": True,
    "performance_monitoring": True
}

TELEGRAM_CONFIG = {
    "token": "7386542811:AAEZ21p30rES1k8NxNM2xbZ53U44PI9D5CY",
    "chat_id": "7590895952",
    "enabled": True
}

# 보안 설정
SECURITY_CONFIG = {
    "webhook_secret": "phoenix_complete_webhook_2025_ultra_secure",
    "api_keys": ["phoenix_complete_key_1", "phoenix_complete_key_2", "phoenix_complete_key_3"],
    "rate_limit_per_minute": 120,
    "max_signal_size": 4096,
    "allowed_ips": [],
    "signature_required": False,
    "session_timeout": 7200,
    "max_request_per_ip": 2000,
    "ddos_protection": True,
    "encryption_enabled": False
}

# 거래 설정
TRADING_CONFIG = {
    "allowed_symbols": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", 
        "XRPUSDT", "SOLUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        "MATICUSDT", "LTCUSDT", "BCHUSDT", "FILUSDT", "TRXUSDT",
        "ATOMUSDT", "NEARUSDT", "SANDUSDT", "MANAUSDT", "FTMUSDT"
    ],
    "min_confidence": 0.25,
    "phoenix_95_threshold": 0.45,
    "max_position_size": 0.15,
    "kelly_fraction": 0.20,
    "quality_threshold": 0.55,
    "real_data_weight": 0.85,
    "phoenix_95_weight": 0.95,
    "performance_adjustment": 0.80,
    "signal_expiry_seconds": 900,
    "risk_management": True,
    "dynamic_sizing": True
}

# 레버리지 및 이솔레이티드 거래 설정
LEVERAGE_CONFIG = {
    "leverage": 20,                    # 20배 레버리지
    "margin_mode": "ISOLATED",         # 이솔레이티드 모드
    "stop_loss_percent": 0.02,         # 2% 손절
    "take_profit_percent": 0.02,       # 2% 익절
    "max_margin_ratio": 0.8,           # 최대 마진 사용률 80%
    "liquidation_buffer": 0.1,         # 청산 방지 버퍼 10%
    "maintenance_margin": 0.004,       # 유지 마진 0.4%
    "trading_fee": 0.0004,             # 거래 수수료 0.04%
    "funding_fee": 0.0001,             # 펀딩 수수료 0.01%
    "max_leverage_symbols": {          # 심볼별 최대 레버리지
        "BTCUSDT": 125,
        "ETHUSDT": 100,
        "BNBUSDT": 50,
        "default": 20
    }
}

# 로깅 설정
def setup_complete_logging():
    """완전한 로깅 설정"""
    log_dir = Path("logs_complete_webhook")
    log_dir.mkdir(exist_ok=True)

    # 기존 로그 파일 백업
    log_files = list(log_dir.glob("*.log"))
    if log_files:
        backup_dir = log_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        for log_file in log_files:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f"{log_file.stem}_{timestamp}.log"
            try:
                log_file.rename(backup_file)
            except:
                pass

    # 로그 포맷터
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | [완성웹훅] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 메인 로거
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러들
    main_handler = logging.FileHandler(
        'logs_complete_webhook/complete_webhook.log', 
        encoding='utf-8', mode='w'
    )
    main_handler.setFormatter(formatter)
    logger.addHandler(main_handler)
    
    signal_handler = logging.FileHandler(
        'logs_complete_webhook/complete_signals.log', 
        encoding='utf-8', mode='w'
    )
    signal_handler.setFormatter(formatter)
    logger.addHandler(signal_handler)
    
    error_handler = logging.FileHandler(
        'logs_complete_webhook/complete_errors.log', 
        encoding='utf-8', mode='w'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

# 텔레그램 전송 함수
async def send_telegram_signal(message: str):
    """텔레그램으로 신호 전송"""
    if not TELEGRAM_CONFIG["enabled"]:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_CONFIG['token']}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CONFIG["chat_id"],
            "text": message,
            "parse_mode": "HTML"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, timeout=10) as response:
                if response.status == 200:
                    logging.info(f"📱 텔레그램 전송 성공")
                else:
                    logging.warning(f"📱 텔레그램 전송 실패: {response.status}")
    except Exception as e:
        logging.error(f"📱 텔레그램 전송 오류: {e}")

# 완성된 신호 검증기
class CompleteSignalValidator:
    def __init__(self):
        self.price_cache = {}
        self.market_data_cache = {}
        self.cache_duration = 90
        self.validation_stats = {
            "total_validated": 0,
            "valid_signals": 0,
            "invalid_signals": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "price_validations": 0,
            "market_validations": 0
        }
        
    async def validate_signal_complete(self, data: Dict) -> Dict:
        """완전한 신호 검증 시스템"""
        validation_result = {
            "valid": False,
            "quality_score": 0.0,
            "errors": [],
            "warnings": [],
            "enhancements": {},
            "validation_time": time.time(),
            "validation_level": "COMPLETE"
        }
        
        try:
            self.validation_stats["total_validated"] += 1
            
            # 1. 기본 구조 검증
            basic_check = self._validate_basic_structure(data)
            if not basic_check["valid"]:
                validation_result["errors"].extend(basic_check["errors"])
                self.validation_stats["invalid_signals"] += 1
                return validation_result
            
            # 2. 심볼 및 거래소 검증
            symbol_check = await self._validate_symbol_complete(data["symbol"])
            validation_result["enhancements"]["symbol_validation"] = symbol_check
            
            # 3. 가격 검증 (실제 바이낸스 데이터)
            price_check = await self._validate_price_complete(data["symbol"], data["price"])
            validation_result["enhancements"]["price_validation"] = price_check
            self.validation_stats["price_validations"] += 1
            
            # 4. 신뢰도 및 전략 검증
            confidence_check = self._validate_confidence_complete(data)
            validation_result["enhancements"]["confidence_analysis"] = confidence_check
            
            # 5. 시장 조건 검증
            market_check = await self._validate_market_conditions_complete(data)
            validation_result["enhancements"]["market_analysis"] = market_check
            self.validation_stats["market_validations"] += 1
            
            # 6. 기술적 지표 검증
            technical_check = self._validate_technical_indicators(data)
            validation_result["enhancements"]["technical_analysis"] = technical_check
            
            # 7. 종합 품질 점수 계산
            quality_score = self._calculate_complete_quality_score(
                data, symbol_check, price_check, confidence_check, 
                market_check, technical_check
            )
            validation_result["quality_score"] = quality_score
            
            # 8. 리스크 평가
            risk_assessment = self._assess_signal_risk(data, validation_result["enhancements"])
            validation_result["enhancements"]["risk_assessment"] = risk_assessment
            
            # 9. 최종 검증 결정
            if (quality_score >= TRADING_CONFIG["quality_threshold"] and 
                not symbol_check.get("errors") and 
                price_check.get("reasonable", False)):
                validation_result["valid"] = True
                self.validation_stats["valid_signals"] += 1
                logging.info(f"신호 검증 성공: {data['symbol']} {data['action']} (품질: {quality_score:.3f})")
            else:
                validation_result["warnings"].append(f"품질 기준 미달: {quality_score:.3f}")
                self.validation_stats["invalid_signals"] += 1
                logging.warning(f"신호 검증 실패: {data['symbol']} {data['action']} (품질: {quality_score:.3f})")
            
        except Exception as e:
            validation_result["errors"].append(f"검증 프로세스 오류: {e}")
            self.validation_stats["invalid_signals"] += 1
            logging.error(f"신호 검증 중 오류: {e}\n{traceback.format_exc()}")
        
        return validation_result
    
    def _validate_basic_structure(self, data: Dict) -> Dict:
        """기본 구조 검증"""
        errors = []
        warnings = []
        
        required_fields = ["symbol", "action", "price"]
        for field in required_fields:
            if field not in data:
                errors.append(f"필수 필드 누락: {field}")
        
        if "symbol" in data:
            if not isinstance(data["symbol"], str):
                errors.append("symbol은 문자열이어야 함")
            elif len(data["symbol"]) < 3:
                errors.append("symbol이 너무 짧음")
        
        if "action" in data:
            action = str(data["action"]).lower()
            if action not in ["buy", "sell", "hold", "long", "short"]:
                errors.append("action은 buy, sell, hold, long, short 중 하나여야 함")
        
        if "price" in data:
            try:
                price = float(data["price"])
                if price <= 0:
                    errors.append("price는 양수여야 함")
                elif price > 1000000:
                    warnings.append("매우 높은 가격")
            except (ValueError, TypeError):
                errors.append("price는 숫자여야 함")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_symbol_complete(self, symbol: str) -> Dict:
        """완전한 심볼 검증"""
        errors = []
        warnings = []
        info = {}
        
        symbol = symbol.upper()
        
        # .P 제거 후 기본 심볼 확인 (선물 심볼 처리)
        base_symbol = symbol.replace('.P', '')
        
        # 허용된 심볼 체크
        if symbol not in TRADING_CONFIG["allowed_symbols"] and base_symbol not in TRADING_CONFIG["allowed_symbols"]:
            if symbol.endswith("USDT") or symbol.endswith("USDT.P") or base_symbol.endswith("USDT"):
                warnings.append(f"비선호 심볼: {symbol}")
            else:
                errors.append(f"허용되지 않은 심볼: {symbol}")
        
        major_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        if symbol in major_symbols or base_symbol in major_symbols:
            info["category"] = "MAJOR"
            info["liquidity"] = "HIGH"
        elif symbol in TRADING_CONFIG["allowed_symbols"][:10] or base_symbol in TRADING_CONFIG["allowed_symbols"][:10]:
            info["category"] = "POPULAR"
            info["liquidity"] = "MEDIUM"
        else:
            info["category"] = "ALTERNATIVE"
            info["liquidity"] = "LOW"
        
        # 선물 심볼 정보 추가
        if ".P" in symbol:
            info["contract_type"] = "FUTURES"
            info["base_symbol"] = base_symbol
        else:
            info["contract_type"] = "SPOT"
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "symbol_normalized": symbol
        }
    
    async def _validate_price_complete(self, symbol: str, signal_price: float) -> Dict:
        """완전한 가격 검증"""
        try:
            cache_key = f"{symbol}_price_complete"
            current_time = time.time()
            
            if cache_key in self.price_cache:
                cached_data, cached_time = self.price_cache[cache_key]
                if current_time - cached_time < self.cache_duration:
                    real_price, volume_24h, price_change_24h = cached_data
                    self.validation_stats["cache_hits"] += 1
                else:
                    real_price, volume_24h, price_change_24h = await self._fetch_complete_market_data(symbol)
            else:
                real_price, volume_24h, price_change_24h = await self._fetch_complete_market_data(symbol)
            
            if real_price:
                price_diff_percent = abs(signal_price - real_price) / real_price * 100
                
                if price_diff_percent < 1.0:
                    reasonableness = "EXCELLENT"
                elif price_diff_percent < 3.0:
                    reasonableness = "GOOD"
                elif price_diff_percent < 5.0:
                    reasonableness = "FAIR"
                elif price_diff_percent < 10.0:
                    reasonableness = "POOR"
                else:
                    reasonableness = "UNACCEPTABLE"
                
                return {
                    "real_price": real_price,
                    "signal_price": signal_price,
                    "difference_percent": round(price_diff_percent, 3),
                    "difference_amount": round(abs(signal_price - real_price), 6),
                    "reasonable": price_diff_percent < 5.0,
                    "reasonableness": reasonableness,
                    "volume_24h": volume_24h,
                    "price_change_24h": price_change_24h,
                    "data_source": "binance_real_complete",
                    "validation_time": current_time
                }
            else:
                return {
                    "real_price": None,
                    "signal_price": signal_price,
                    "difference_percent": 0,
                    "reasonable": True,
                    "reasonableness": "UNKNOWN",
                    "data_source": "api_failed",
                    "warning": "실제 가격 조회 실패"
                }
                
        except Exception as e:
            logging.warning(f"완전한 가격 검증 실패: {e}")
            return {
                "real_price": None,
                "signal_price": signal_price,
                "reasonable": True,
                "data_source": "error",
                "error": str(e)
            }
    
    async def _fetch_complete_market_data(self, symbol: str) -> tuple:
        """완전한 시장 데이터 조회"""
        try:
            self.validation_stats["api_calls"] += 1
            
            stats_url = "https://api.binance.com/api/v3/ticker/24hr"
            params = {"symbol": symbol}
            
            response = await asyncio.wait_for(
                asyncio.to_thread(requests.get, stats_url, params=params, timeout=10),
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                price = float(data["lastPrice"])
                volume_24h = float(data["volume"])
                price_change_24h = float(data["priceChangePercent"])
                
                cache_key = f"{symbol}_price_complete"
                self.price_cache[cache_key] = ((price, volume_24h, price_change_24h), time.time())
                
                return price, volume_24h, price_change_24h
            
        except Exception as e:
            logging.warning(f"완전한 시장 데이터 조회 실패 ({symbol}): {e}")
        
        return None, None, None
    
    def _validate_confidence_complete(self, data: Dict) -> Dict:
        """완전한 신뢰도 검증"""
        confidence = data.get("confidence", 0.8)
        
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
            
            if confidence >= 0.95:
                grade = "EXCEPTIONAL"
                risk_level = "VERY_LOW"
            elif confidence >= 0.85:
                grade = "EXCELLENT"
                risk_level = "LOW"
            elif confidence >= 0.75:
                grade = "GOOD"
                risk_level = "MEDIUM_LOW"
            elif confidence >= 0.65:
                grade = "FAIR"
                risk_level = "MEDIUM"
            elif confidence >= 0.50:
                grade = "POOR"
                risk_level = "MEDIUM_HIGH"
            elif confidence >= 0.35:
                grade = "VERY_POOR"
                risk_level = "HIGH"
            else:
                grade = "UNACCEPTABLE"
                risk_level = "VERY_HIGH"
            
            strategy = data.get("strategy", "unknown")
            strategy_confidence = self._analyze_strategy_reliability(strategy)
            
            return {
                "confidence": confidence,
                "grade": grade,
                "risk_level": risk_level,
                "strategy": strategy,
                "strategy_confidence": strategy_confidence,
                "min_threshold": TRADING_CONFIG["min_confidence"],
                "phoenix_95_threshold": TRADING_CONFIG["phoenix_95_threshold"],
                "passes_min": confidence >= TRADING_CONFIG["min_confidence"],
                "qualifies_for_phoenix_95": confidence >= TRADING_CONFIG["phoenix_95_threshold"],
                "recommended_position_size": min(confidence * 0.2, TRADING_CONFIG["max_position_size"])
            }
            
        except (ValueError, TypeError):
            return {
                "confidence": 0.0,
                "grade": "INVALID",
                "risk_level": "MAXIMUM",
                "passes_min": False,
                "qualifies_for_phoenix_95": False,
                "error": "잘못된 신뢰도 값"
            }
    
    def _analyze_strategy_reliability(self, strategy: str) -> Dict:
        """전략 신뢰성 분석"""
        strategy_ratings = {
            "momentum": {"reliability": 0.75, "volatility": "MEDIUM"},
            "breakout": {"reliability": 0.70, "volatility": "HIGH"},
            "mean_reversion": {"reliability": 0.65, "volatility": "MEDIUM"},
            "trend_following": {"reliability": 0.80, "volatility": "LOW"},
            "scalping": {"reliability": 0.60, "volatility": "VERY_HIGH"},
            "swing": {"reliability": 0.75, "volatility": "MEDIUM"},
            "day_trading": {"reliability": 0.65, "volatility": "HIGH"},
            "unknown": {"reliability": 0.50, "volatility": "UNKNOWN"}
        }
        
        strategy_lower = str(strategy).lower()
        for key, value in strategy_ratings.items():
            if key in strategy_lower:
                return {
                    "strategy_type": key,
                    "reliability": value["reliability"],
                    "expected_volatility": value["volatility"],
                    "confidence_multiplier": value["reliability"]
                }
        
        return strategy_ratings["unknown"]
    
    async def _validate_market_conditions_complete(self, data: Dict) -> Dict:
        """완전한 시장 조건 검증"""
        try:
            symbol = data["symbol"]
            current_time = datetime.now()
            current_hour = current_time.hour
            weekday = current_time.weekday()
            
            market_session = self._analyze_market_session(current_hour)
            volume_analysis = await self._analyze_volume_conditions(symbol)
            volatility_analysis = await self._analyze_volatility_conditions(symbol)
            
            market_score = self._calculate_market_score(
                market_session, volume_analysis, volatility_analysis, weekday
            )
            
            return {
                "current_time": current_time.isoformat(),
                "market_session": market_session,
                "is_weekend": weekday >= 5,
                "volume_analysis": volume_analysis,
                "volatility_analysis": volatility_analysis,
                "market_score": market_score,
                "trading_favorable": market_score > 0.6,
                "optimal_timing": market_score > 0.8,
                "risk_factors": self._identify_complete_risk_factors(
                    data, market_session, weekday, volume_analysis, volatility_analysis
                ),
                "recommendations": self._generate_timing_recommendations(market_score, market_session)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "trading_favorable": True,
                "market_score": 0.5
            }
    
    def _analyze_market_session(self, hour: int) -> Dict:
        """시장 세션 분석"""
        sessions = {
            "ASIA_NIGHT": (0, 6, 0.3),
            "ASIA_MORNING": (6, 9, 0.7),
            "ASIA_ACTIVE": (9, 15, 0.9),
            "EUROPE_OPENING": (15, 18, 0.8),
            "EUROPE_ACTIVE": (18, 21, 0.9),
            "US_OPENING": (21, 24, 0.85)
        }
        
        for session_name, (start, end, activity) in sessions.items():
            if start <= hour < end:
                return {
                    "name": session_name,
                    "activity_level": activity,
                    "description": f"{session_name.replace('_', ' ').title()}"
                }
        
        return {"name": "UNKNOWN", "activity_level": 0.5, "description": "Unknown Session"}
    
    async def _analyze_volume_conditions(self, symbol: str) -> Dict:
        """거래량 조건 분석"""
        try:
            cache_key = f"{symbol}_volume_analysis"
            if cache_key in self.market_data_cache:
                cached_data, cached_time = self.market_data_cache[cache_key]
                if time.time() - cached_time < 300:
                    return cached_data
            
            _, volume_24h, _ = await self._fetch_complete_market_data(symbol)
            
            if volume_24h:
                volume_thresholds = {
                    "BTCUSDT": 50000, "ETHUSDT": 200000, "BNBUSDT": 10000
                }
                threshold = volume_thresholds.get(symbol, 5000)
                
                if volume_24h > threshold * 2:
                    level = "VERY_HIGH"
                    score = 0.95
                elif volume_24h > threshold * 1.5:
                    level = "HIGH"
                    score = 0.85
                elif volume_24h > threshold:
                    level = "NORMAL"
                    score = 0.70
                elif volume_24h > threshold * 0.5:
                    level = "LOW"
                    score = 0.50
                else:
                    level = "VERY_LOW"
                    score = 0.30
                
                result = {
                    "volume_24h": volume_24h,
                    "level": level,
                    "score": score,
                    "threshold": threshold,
                    "liquidity": "EXCELLENT" if score > 0.8 else "GOOD" if score > 0.6 else "POOR"
                }
                
                self.market_data_cache[cache_key] = (result, time.time())
                return result
            
        except Exception as e:
            logging.warning(f"거래량 분석 실패: {e}")
        
        return {"level": "UNKNOWN", "score": 0.5, "liquidity": "UNKNOWN"}
    
    async def _analyze_volatility_conditions(self, symbol: str) -> Dict:
        """변동성 조건 분석"""
        try:
            _, _, price_change_24h = await self._fetch_complete_market_data(symbol)
            
            if price_change_24h is not None:
                abs_change = abs(price_change_24h)
                
                if abs_change > 15:
                    level = "EXTREME"
                    score = 0.3
                elif abs_change > 10:
                    level = "VERY_HIGH"
                    score = 0.4
                elif abs_change > 7:
                    level = "HIGH"
                    score = 0.6
                elif abs_change > 4:
                    level = "MEDIUM"
                    score = 0.8
                elif abs_change > 2:
                    level = "LOW"
                    score = 0.9
                else:
                    level = "VERY_LOW"
                    score = 0.7
                
                return {
                    "price_change_24h": price_change_24h,
                    "abs_change": abs_change,
                    "level": level,
                    "score": score,
                    "direction": "UP" if price_change_24h > 0 else "DOWN" if price_change_24h < 0 else "FLAT",
                    "opportunity": "EXCELLENT" if 2 <= abs_change <= 7 else "GOOD" if abs_change <= 10 else "POOR"
                }
            
        except Exception as e:
            logging.warning(f"변동성 분석 실패: {e}")
        
        return {"level": "UNKNOWN", "score": 0.5, "opportunity": "UNKNOWN"}
    
    def _calculate_market_score(self, session: Dict, volume: Dict, volatility: Dict, weekday: int) -> float:
        """종합 시장 점수 계산"""
        session_score = session.get("activity_level", 0.5)
        volume_score = volume.get("score", 0.5)
        volatility_score = volatility.get("score", 0.5)
        
        weekend_penalty = 0.8 if weekday >= 5 else 1.0
        
        market_score = (
            session_score * 0.3 +
            volume_score * 0.4 +
            volatility_score * 0.3
        ) * weekend_penalty
        
        return round(min(max(market_score, 0.0), 1.0), 3)
    
    def _identify_complete_risk_factors(self, data: Dict, session: Dict, weekday: int, 
                                      volume: Dict, volatility: Dict) -> List[str]:
        """완전한 리스크 요인 식별"""
        risk_factors = []
        
        if weekday >= 5:
            risk_factors.append("주말 거래 (낮은 유동성)")
        
        if session.get("activity_level", 0.5) < 0.5:
            risk_factors.append("비활성 시간대")
        
        confidence = data.get("confidence", 0.8)
        if confidence < 0.5:
            risk_factors.append("낮은 신호 신뢰도")
        
        if volume.get("score", 0.5) < 0.4:
            risk_factors.append("낮은 거래량")
        
        volatility_level = volatility.get("level", "UNKNOWN")
        if volatility_level in ["EXTREME", "VERY_HIGH"]:
            risk_factors.append("과도한 변동성")
        elif volatility_level == "VERY_LOW":
            risk_factors.append("낮은 변동성 (기회 부족)")
        
        symbol = data["symbol"].upper()
        if symbol not in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            risk_factors.append("주요 심볼이 아님")
        
        return risk_factors
    
    def _generate_timing_recommendations(self, market_score: float, session: Dict) -> List[str]:
        """타이밍 추천사항 생성"""
        recommendations = []
        
        if market_score > 0.8:
            recommendations.append("최적의 거래 타이밍")
        elif market_score > 0.6:
            recommendations.append("좋은 거래 타이밍")
        elif market_score > 0.4:
            recommendations.append("주의 깊은 거래 권장")
        else:
            recommendations.append("거래 대기 권장")
        
        session_name = session.get("name", "")
        if session_name in ["ASIA_ACTIVE", "EUROPE_ACTIVE"]:
            recommendations.append("활성 세션 - 적극적 거래 가능")
        elif session_name in ["ASIA_NIGHT"]:
            recommendations.append("저활성 세션 - 보수적 접근")
        
        return recommendations
    
    def _validate_technical_indicators(self, data: Dict) -> Dict:
        """기술적 지표 검증"""
        indicators = {}
        score = 0.5
        
        if "rsi" in data:
            try:
                rsi = float(data["rsi"])
                if 0 <= rsi <= 100:
                    indicators["rsi"] = {
                        "value": rsi,
                        "valid": True,
                        "signal": "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL",
                        "strength": abs(rsi - 50) / 50
                    }
                    score += 0.1
                else:
                    indicators["rsi"] = {"value": rsi, "valid": False, "error": "범위 초과"}
            except (ValueError, TypeError):
                indicators["rsi"] = {"valid": False, "error": "잘못된 형식"}
        
        if "macd" in data:
            try:
                macd = float(data["macd"])
                indicators["macd"] = {
                    "value": macd,
                    "valid": True,
                    "signal": "BULLISH" if macd > 0 else "BEARISH",
                    "strength": min(abs(macd) * 10, 1.0)
                }
                score += 0.1
            except (ValueError, TypeError):
                indicators["macd"] = {"valid": False, "error": "잘못된 형식"}
        
        if "volume" in data:
            try:
                volume = float(data["volume"])
                if volume >= 0:
                    indicators["volume"] = {
                        "value": volume,
                        "valid": True,
                        "level": "HIGH" if volume > 1000000 else "MEDIUM" if volume > 100000 else "LOW"
                    }
                    score += 0.05
            except (ValueError, TypeError):
                indicators["volume"] = {"valid": False, "error": "잘못된 형식"}
        
        if "timeframe" in data:
            timeframe = str(data["timeframe"]).lower()
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if timeframe in valid_timeframes:
                indicators["timeframe"] = {
                    "value": timeframe,
                    "valid": True,
                    "reliability": {"1m": 0.3, "5m": 0.5, "15m": 0.6, "30m": 0.7, 
                                  "1h": 0.8, "4h": 0.9, "1d": 0.95}.get(timeframe, 0.5)
                }
                score += 0.05
        
        return {
            "indicators": indicators,
            "score": min(score, 1.0),
            "indicator_count": len([i for i in indicators.values() if i.get("valid", False)])
        }
    
    def _assess_signal_risk(self, data: Dict, enhancements: Dict) -> Dict:
        """신호 리스크 평가"""
        risk_score = 0.5
        risk_factors = []
        
        confidence = data.get("confidence", 0.8)
        if confidence < 0.4:
            risk_score += 0.3
            risk_factors.append("매우 낮은 신뢰도")
        elif confidence < 0.6:
            risk_score += 0.1
            risk_factors.append("낮은 신뢰도")
        
        price_validation = enhancements.get("price_validation", {})
        if price_validation.get("difference_percent", 0) > 5:
            risk_score += 0.2
            risk_factors.append("가격 불일치")
        
        market_analysis = enhancements.get("market_analysis", {})
        if not market_analysis.get("trading_favorable", True):
            risk_score += 0.15
            risk_factors.append("불리한 시장 조건")
        
        volatility = market_analysis.get("volatility_analysis", {})
        if volatility.get("level") in ["EXTREME", "VERY_HIGH"]:
            risk_score += 0.2
            risk_factors.append("과도한 변동성")
        
        volume = market_analysis.get("volume_analysis", {})
        if volume.get("score", 0.5) < 0.4:
            risk_score += 0.1
            risk_factors.append("낮은 거래량")
        
        risk_score = min(max(risk_score, 0.0), 1.0)
        
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.5:
            risk_level = "MEDIUM_LOW"
        elif risk_score < 0.7:
            risk_level = "MEDIUM"
        elif risk_score < 0.85:
            risk_level = "MEDIUM_HIGH"
        else:
            risk_level = "HIGH"
        
        return {
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommended_position_size": max(0.01, (1 - risk_score) * TRADING_CONFIG["max_position_size"]),
            "risk_adjusted_confidence": max(0.1, confidence * (1 - risk_score))
        }
    
    def _calculate_complete_quality_score(self, data: Dict, symbol_check: Dict, 
                                        price_check: Dict, confidence_check: Dict, 
                                        market_check: Dict, technical_check: Dict) -> float:
        """완전한 품질 점수 계산"""
        score = 0.0
        
        confidence = confidence_check.get("confidence", 0.0)
        score += confidence * 0.3
        
        if price_check.get("reasonable", False):
            score += 0.2
            diff_pct = price_check.get("difference_percent", 100)
            if diff_pct < 1:
                score += 0.05
            elif diff_pct < 2:
                score += 0.03
        
        symbol_info = symbol_check.get("info", {})
        if symbol_info.get("category") == "MAJOR":
            score += 0.15
        elif symbol_info.get("category") == "POPULAR":
            score += 0.12
        elif symbol_info.get("category") == "ALTERNATIVE":
            score += 0.08
        
        market_score = market_check.get("market_score", 0.5)
        score += market_score * 0.2
        
        technical_score = technical_check.get("score", 0.5)
        score += technical_score * 0.1
        
        strategy_confidence = confidence_check.get("strategy_confidence", {}).get("reliability", 0.5)
        score += strategy_confidence * 0.05
        
        volume_analysis = market_check.get("volume_analysis", {})
        if volume_analysis.get("level") in ["HIGH", "VERY_HIGH"]:
            score += 0.02
        
        if market_check.get("optimal_timing", False):
            score += 0.03
        
        indicator_count = technical_check.get("indicator_count", 0)
        if indicator_count >= 3:
            score += 0.02
        elif indicator_count >= 2:
            score += 0.01
        
        risk_factors = market_check.get("risk_factors", [])
        penalty = min(len(risk_factors) * 0.02, 0.1)
        score -= penalty
        
        return round(min(max(score, 0.0), 1.0), 4)

# Phoenix 95점 완전 신호 분석기
class Phoenix95CompleteAnalyzer:
    def __init__(self):
        self.analysis_cache = {}
        self.cache_duration = 120
        self.performance_history = deque(maxlen=1000)
        
        if PHOENIX_95_AVAILABLE:
            try:
                self.phoenix_95_engine = Phoenix95Engine(initial_balance=10000)
                self.phoenix_95_ready = True
                logging.info("✅ Phoenix 95점 엔진 완전 초기화 성공")
            except Exception as e:
                self.phoenix_95_ready = False
                logging.error(f"❌ Phoenix 95점 엔진 초기화 실패: {e}")
        else:
            self.phoenix_95_ready = False
            logging.warning("⚠️ Phoenix 95점 엔진 미사용 - 기본 분석 모드")
        
        self.analysis_stats = {
            "total_analyses": 0,
            "phoenix_95_analyses": 0,
            "enhanced_analyses": 0,
            "emergency_analyses": 0,
            "avg_processing_time": 0,
            "cache_hit_rate": 0
        }
        
    async def analyze_signal_phoenix_95_complete(self, signal_data: Dict, validation_result: Dict) -> Dict:
        """Phoenix 95점 완전 신호 분석"""
        start_time = time.time()
        
        try:
            self.analysis_stats["total_analyses"] += 1
            
            cache_key = self._generate_analysis_cache_key(signal_data)
            if cache_key in self.analysis_cache:
                cached_data, cached_time = self.analysis_cache[cache_key]
                if time.time() - cached_time < self.cache_duration:
                    self.analysis_stats["cache_hit_rate"] = len([k for k, (_, t) in self.analysis_cache.items() if time.time() - t < self.cache_duration]) / len(self.analysis_cache)
                    cached_data["from_cache"] = True
                    return cached_data
            
            quality_score = validation_result["quality_score"]
            confidence = signal_data.get("confidence", 0.8)
            
            if (quality_score >= TRADING_CONFIG["phoenix_95_threshold"] and 
                confidence >= TRADING_CONFIG["phoenix_95_threshold"] and 
                self.phoenix_95_ready):
                analysis = await self._phoenix_95_full_analysis(signal_data, validation_result)
                self.analysis_stats["phoenix_95_analyses"] += 1
            elif quality_score >= TRADING_CONFIG["quality_threshold"]:
                analysis = await self._enhanced_complete_analysis(signal_data, validation_result)
                self.analysis_stats["enhanced_analyses"] += 1
            else:
                analysis = self._basic_complete_analysis(signal_data, validation_result)
            
            if TRADING_CONFIG["risk_management"]:
                analysis = self._apply_risk_management(analysis)
            
            if TRADING_CONFIG["dynamic_sizing"]:
                analysis = self._apply_dynamic_sizing(analysis, validation_result)
            
            analysis = self._apply_kelly_formula_complete(analysis)
            analysis = self._apply_performance_adjustment_complete(analysis)
            analysis = self._final_analysis_validation(analysis, validation_result)
            
            processing_time = time.time() - start_time
            analysis["processing_time"] = round(processing_time, 4)
            
            self._update_analysis_stats(processing_time)
            
            self.analysis_cache[cache_key] = (analysis, time.time())
            
            self.performance_history.append({
                "timestamp": time.time(),
                "quality_score": quality_score,
                "final_confidence": analysis["final_confidence"],
                "analysis_type": analysis["analysis_type"],
                "processing_time": processing_time
            })
            
            logging.info(
                f"완전 분석 완료: {signal_data['symbol']} {signal_data['action']} "
                f"품질:{quality_score:.3f} 신뢰도:{analysis['final_confidence']:.3f} "
                f"타입:{analysis['analysis_type']} 시간:{processing_time:.3f}s"
            )
            
            return analysis
            
        except Exception as e:
            self.analysis_stats["emergency_analyses"] += 1
            logging.error(f"완전 분석 중 오류: {e}\n{traceback.format_exc()}")
            return self._emergency_complete_analysis(signal_data, str(e))
    
    def _generate_analysis_cache_key(self, signal_data: Dict) -> str:
        """분석 캐시 키 생성"""
        symbol = signal_data["symbol"]
        action = signal_data["action"]
        price_band = int(float(signal_data["price"]) / 100) * 100
        confidence_band = int(signal_data.get("confidence", 0.8) * 10)
        time_band = int(time.time() / self.cache_duration) * self.cache_duration
        strategy = signal_data.get("strategy", "default")
        
        cache_string = f"{symbol}_{action}_{price_band}_{confidence_band}_{time_band}_{strategy}"
        return hashlib.md5(cache_string.encode()).hexdigest()[:16]
    
    async def _phoenix_95_full_analysis(self, signal_data: Dict, validation_result: Dict) -> Dict:
        """Phoenix 95점 풀 분석"""
        try:
            phoenix_95_result = self.phoenix_95_engine.analyze_with_95_score(signal_data)
            
            original_confidence = signal_data.get("confidence", 0.8)
            quality_score = validation_result["quality_score"]
            phoenix_95_score = phoenix_95_result.get("ai_score", 50) / 100
            
            market_regime = phoenix_95_result.get("market_regime", {})
            regime_confidence = market_regime.get("confidence_score", 0.5)
            regime_stability = market_regime.get("regime_stability", 0.5)
            
            position_sizing = phoenix_95_result.get("position_sizing", {})
            kelly_ratio = position_sizing.get("kelly_ratio", 0.05)
            
            exit_strategy = phoenix_95_result.get("exit_strategy", {})
            backtest_simulation = phoenix_95_result.get("backtest_simulation", {})
            
            final_confidence = (
                original_confidence * 0.15 +
                quality_score * 0.25 +
                phoenix_95_score * 0.35 +
                regime_confidence * 0.15 +
                regime_stability * 0.10
            ) * TRADING_CONFIG["phoenix_95_weight"]
            
            risk_assessment = validation_result.get("enhancements", {}).get("risk_assessment", {})
            risk_score = risk_assessment.get("risk_score", 0.5)
            risk_adjusted_confidence = final_confidence * (1 - risk_score * 0.5)
            
            # 레버리지 및 이솔레이티드 계산
            leverage_analysis = self._calculate_leverage_position(signal_data, final_confidence)
            
            # 2% 익절/손절 설정
            action = signal_data["action"].lower()
            if action in ["buy", "long"]:
                stop_loss = 1.0 - LEVERAGE_CONFIG["stop_loss_percent"]      # 98% (2% 손절)
                take_profit = 1.0 + LEVERAGE_CONFIG["take_profit_percent"]  # 102% (2% 익절)
            else:
                stop_loss = 1.0 + LEVERAGE_CONFIG["stop_loss_percent"]      # 102% (2% 손절)
                take_profit = 1.0 - LEVERAGE_CONFIG["take_profit_percent"]  # 98% (2% 익절)
            
            return {
                "analysis_type": "PHOENIX_95_COMPLETE_FULL",
                "analysis_version": "4.0.0-complete-leverage",
                "original_confidence": original_confidence,
                "quality_score": quality_score,
                "phoenix_95_score": phoenix_95_score,
                "final_confidence": min(1.0, final_confidence),
                "risk_adjusted_confidence": min(1.0, risk_adjusted_confidence),
                
                "market_regime": market_regime,
                "regime_confidence": regime_confidence,
                "regime_stability": regime_stability,
                
                "position_sizing": position_sizing,
                "kelly_ratio": kelly_ratio,
                "recommended_position_size": leverage_analysis["base_position_size"],
                
                # 레버리지 및 이솔레이티드 정보
                "leverage_analysis": leverage_analysis,
                "leverage": leverage_analysis["leverage"],
                "margin_mode": "ISOLATED",
                "actual_position_size": leverage_analysis["actual_position_size"],
                "margin_required": leverage_analysis["margin_required"],
                "margin_ratio": leverage_analysis["margin_ratio"],
                "liquidation_price": leverage_analysis["liquidation_price"],
                
                "risk_assessment": risk_assessment,
                "risk_level": risk_assessment.get("risk_level", "MEDIUM"),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "stop_loss_percent": LEVERAGE_CONFIG["stop_loss_percent"] * 100,
                "take_profit_percent": LEVERAGE_CONFIG["take_profit_percent"] * 100,
                "risk_reward_ratio": 1.0,  # 2%:2% = 1:1
                
                "execution_timing": self._determine_execution_timing(final_confidence, risk_score),
                "urgency": min(10, int(final_confidence * 10)),
                "market_timing": "EXCELLENT" if phoenix_95_score > 0.8 else "GOOD" if phoenix_95_score > 0.6 else "FAIR",
                
                "key_factors": [
                    f"Phoenix 95점 분석: {phoenix_95_score:.3f}",
                    f"레버리지: {leverage_analysis['leverage']}x",
                    f"마진 모드: ISOLATED",
                    f"실제 포지션: ${leverage_analysis['actual_position_size']:,.2f}",
                    f"필요 마진: ${leverage_analysis['margin_required']:,.2f}",
                    f"청산가: ${leverage_analysis['liquidation_price']:,.2f}"
                ],
                "confidence_breakdown": {
                    "original": original_confidence,
                    "quality": quality_score,
                    "ai_score": phoenix_95_score,
                    "regime": regime_confidence,
                    "stability": regime_stability
                },
                
                "backtest_simulation": backtest_simulation,
                "expected_performance": phoenix_95_result.get("expected_profit_pct", 0),
                
                "phoenix_95_features": [
                    "realistic_backtesting",
                    "smart_position_sizing", 
                    "dynamic_exit_strategy",
                    "market_regime_detection",
                    "isolated_margin_trading",
                    "20x_leverage_support"
                ],
                "validation_data": validation_result,
                "phoenix_95_details": phoenix_95_result,
                "timestamp": time.time(),
                "from_cache": False
            }
            
        except Exception as e:
            logging.error(f"Phoenix 95점 풀 분석 실패: {e}")
            return await self._enhanced_complete_analysis(signal_data, validation_result)
    
    def _calculate_leverage_position(self, signal_data: Dict, confidence: float) -> Dict:
        """레버리지 포지션 계산"""
        try:
            symbol = signal_data["symbol"].replace('.P', '')  # 선물 심볼 처리
            price = float(signal_data["price"])
            action = signal_data["action"].lower()
            
            # 심볼별 최대 레버리지 확인
            max_leverage = LEVERAGE_CONFIG["max_leverage_symbols"].get(symbol, 
                         LEVERAGE_CONFIG["max_leverage_symbols"]["default"])
            leverage = min(LEVERAGE_CONFIG["leverage"], max_leverage)
            
            # 기본 포지션 크기 계산 (신뢰도 기반)
            base_position_ratio = min(confidence * 0.15, TRADING_CONFIG["max_position_size"])
            
            # 가상 자본금 (시뮬레이션)
            virtual_capital = 10000  # $10,000
            base_position_size = virtual_capital * base_position_ratio
            
            # 레버리지 적용한 실제 포지션 크기
            actual_position_size = base_position_size * leverage
            
            # 필요 마진 계산
            margin_required = actual_position_size / leverage
            
            # 마진 비율 계산
            margin_ratio = margin_required / virtual_capital
            
            # 청산가 계산
            liquidation_price = self._calculate_liquidation_price(
                price, action, leverage, margin_ratio
            )
            
            # 예상 수익/손실 계산 (2% 익절/손절 기준)
            profit_loss_2pct = actual_position_size * LEVERAGE_CONFIG["take_profit_percent"]
            
            # 리스크 평가
            risk_level = self._assess_leverage_risk(leverage, margin_ratio, confidence)
            
            return {
                "symbol": symbol,
                "leverage": leverage,
                "max_available_leverage": max_leverage,
                "base_position_size": base_position_size,
                "actual_position_size": actual_position_size,
                "margin_required": margin_required,
                "margin_ratio": margin_ratio,
                "virtual_capital": virtual_capital,
                "liquidation_price": liquidation_price,
                "profit_loss_2pct": profit_loss_2pct,
                "risk_level": risk_level,
                "margin_mode": "ISOLATED",
                "maintenance_margin": LEVERAGE_CONFIG["maintenance_margin"],
                "trading_fee": LEVERAGE_CONFIG["trading_fee"],
                "funding_fee": LEVERAGE_CONFIG["funding_fee"]
            }
            
        except Exception as e:
            logging.warning(f"레버리지 포지션 계산 실패: {e}")
            return {
                "leverage": 1,
                "base_position_size": 100,
                "actual_position_size": 100,
                "margin_required": 100,
                "margin_ratio": 0.01,
                "liquidation_price": 0,
                "error": str(e)
            }
    
    def _calculate_liquidation_price(self, entry_price: float, action: str, leverage: int, margin_ratio: float) -> float:
        """청산가 계산"""
        try:
            # 유지 마진 + 버퍼
            maintenance_margin_rate = LEVERAGE_CONFIG["maintenance_margin"]
            liquidation_buffer = LEVERAGE_CONFIG["liquidation_buffer"]
            
            if action in ["buy", "long"]:
                # 롱 포지션 청산가
                liquidation_price = entry_price * (1 - (1/leverage - maintenance_margin_rate - liquidation_buffer))
            else:
                # 숏 포지션 청산가  
                liquidation_price = entry_price * (1 + (1/leverage - maintenance_margin_rate - liquidation_buffer))
            
            return max(0, liquidation_price)
            
        except Exception as e:
            logging.warning(f"청산가 계산 실패: {e}")
            return 0
    
    def _assess_leverage_risk(self, leverage: int, margin_ratio: float, confidence: float) -> str:
        """레버리지 리스크 평가"""
        risk_score = 0
        
        # 레버리지 리스크
        if leverage >= 50:
            risk_score += 3
        elif leverage >= 20:
            risk_score += 2
        elif leverage >= 10:
            risk_score += 1
        
        # 마진 비율 리스크
        if margin_ratio > 0.5:
            risk_score += 3
        elif margin_ratio > 0.3:
            risk_score += 2
        elif margin_ratio > 0.1:
            risk_score += 1
        
        # 신뢰도 리스크
        if confidence < 0.5:
            risk_score += 2
        elif confidence < 0.7:
            risk_score += 1
        
        if risk_score >= 6:
            return "EXTREME"
        elif risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def _enhanced_complete_analysis(self, signal_data: Dict, validation_result: Dict) -> Dict:
        """강화된 완전 분석"""
        try:
            original_confidence = signal_data.get("confidence", 0.8)
            quality_score = validation_result["quality_score"]
            
            market_analysis = validation_result.get("enhancements", {}).get("market_analysis", {})
            market_score = market_analysis.get("market_score", 0.5)
            
            price_validation = validation_result.get("enhancements", {}).get("price_validation", {})
            price_accuracy = 1.0 - (price_validation.get("difference_percent", 0) / 100)
            
            confidence_analysis = validation_result.get("enhancements", {}).get("confidence_analysis", {})
            strategy_confidence = confidence_analysis.get("strategy_confidence", {}).get("reliability", 0.5)
            
            technical_analysis = validation_result.get("enhancements", {}).get("technical_analysis", {})
            technical_score = technical_analysis.get("score", 0.5)
            
            final_confidence = (
                original_confidence * 0.25 +
                quality_score * 0.30 +
                market_score * 0.20 +
                price_accuracy * 0.15 +
                strategy_confidence * 0.10
            ) * TRADING_CONFIG["real_data_weight"]
            
            # 레버리지 및 이솔레이티드 계산
            leverage_analysis = self._calculate_leverage_position(signal_data, final_confidence)
            
            # 2% 익절/손절 설정
            action = signal_data["action"].lower()
            if action in ["buy", "long"]:
                stop_loss = 1.0 - LEVERAGE_CONFIG["stop_loss_percent"]      # 98%
                take_profit = 1.0 + LEVERAGE_CONFIG["take_profit_percent"]  # 102%
            else:
                stop_loss = 1.0 + LEVERAGE_CONFIG["stop_loss_percent"]      # 102%
                take_profit = 1.0 - LEVERAGE_CONFIG["take_profit_percent"]  # 98%
            
            return {
                "analysis_type": "ENHANCED_COMPLETE_LEVERAGE",
                "analysis_version": "4.0.0-enhanced-leverage",
                "original_confidence": original_confidence,
                "quality_score": quality_score,
                "final_confidence": min(1.0, final_confidence),
                
                "market_score": market_score,
                "price_accuracy": price_accuracy,
                "strategy_confidence": strategy_confidence,
                "technical_score": technical_score,
                
                "recommended_position_size": leverage_analysis["base_position_size"],
                "position_rationale": "품질 및 시장 조건 기반 + 레버리지",
                
                # 레버리지 및 이솔레이티드 정보
                "leverage_analysis": leverage_analysis,
                "leverage": leverage_analysis["leverage"],
                "margin_mode": "ISOLATED",
                "actual_position_size": leverage_analysis["actual_position_size"],
                "margin_required": leverage_analysis["margin_required"],
                "margin_ratio": leverage_analysis["margin_ratio"],
                "liquidation_price": leverage_analysis["liquidation_price"],
                
                "risk_level": "MEDIUM_LOW" if final_confidence > 0.7 else "MEDIUM" if final_confidence > 0.5 else "HIGH",
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "stop_loss_percent": LEVERAGE_CONFIG["stop_loss_percent"] * 100,
                "take_profit_percent": LEVERAGE_CONFIG["take_profit_percent"] * 100,
                "risk_reward_ratio": 1.0,
                
                "execution_timing": "IMMEDIATE" if final_confidence > 0.75 else "CAREFUL" if final_confidence > 0.6 else "DELAYED",
                "urgency": min(8, int(final_confidence * 10)),
                "market_timing": "GOOD" if market_score > 0.7 else "FAIR",
                
                "key_factors": [
                    f"강화 분석 완료",
                    f"레버리지: {leverage_analysis['leverage']}x",
                    f"마진 모드: ISOLATED", 
                    f"실제 포지션: ${leverage_analysis['actual_position_size']:,.2f}",
                    f"필요 마진: ${leverage_analysis['margin_required']:,.2f}"
                ],
                "confidence_breakdown": {
                    "original": original_confidence,
                    "quality": quality_score,
                    "market": market_score,
                    "price": price_accuracy,
                    "strategy": strategy_confidence
                },
                
                "validation_data": validation_result,
                "timestamp": time.time(),
                "from_cache": False
            }
            
        except Exception as e:
            logging.error(f"강화된 완전 분석 실패: {e}")
            return self._basic_complete_analysis(signal_data, validation_result)
    
    def _basic_complete_analysis(self, signal_data: Dict, validation_result: Dict) -> Dict:
        """기본 완전 분석"""
        original_confidence = signal_data.get("confidence", 0.8)
        quality_score = validation_result["quality_score"]
        
        final_confidence = (original_confidence + quality_score) / 2 * 0.85
        
        # 기본 레버리지 계산
        leverage_analysis = self._calculate_leverage_position(signal_data, final_confidence)
        
        # 2% 익절/손절 설정
        action = signal_data["action"].lower()
        if action in ["buy", "long"]:
            stop_loss = 1.0 - LEVERAGE_CONFIG["stop_loss_percent"]      # 98%
            take_profit = 1.0 + LEVERAGE_CONFIG["take_profit_percent"]  # 102%
        else:
            stop_loss = 1.0 + LEVERAGE_CONFIG["stop_loss_percent"]      # 102%
            take_profit = 1.0 - LEVERAGE_CONFIG["take_profit_percent"]  # 98%
        
        return {
            "analysis_type": "BASIC_COMPLETE_LEVERAGE",
            "analysis_version": "4.0.0-basic-leverage",
            "original_confidence": original_confidence,
            "quality_score": quality_score,
            "final_confidence": min(1.0, final_confidence),
            
            "recommended_position_size": leverage_analysis["base_position_size"],
            
            # 레버리지 및 이솔레이티드 정보
            "leverage_analysis": leverage_analysis,
            "leverage": leverage_analysis["leverage"],
            "margin_mode": "ISOLATED",
            "actual_position_size": leverage_analysis["actual_position_size"],
            "margin_required": leverage_analysis["margin_required"],
            "liquidation_price": leverage_analysis["liquidation_price"],
            
            "risk_level": "MEDIUM",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "stop_loss_percent": LEVERAGE_CONFIG["stop_loss_percent"] * 100,
            "take_profit_percent": LEVERAGE_CONFIG["take_profit_percent"] * 100,
            
            "execution_timing": "CAREFUL",
            "urgency": min(5, int(final_confidence * 8)),
            "market_timing": "FAIR",
            
            "key_factors": [
                "기본 분석 완료",
                f"레버리지: {leverage_analysis['leverage']}x",
                f"마진 모드: ISOLATED",
                f"보수적 접근",
                f"품질 점수: {quality_score:.3f}"
            ],
            
            "validation_data": validation_result,
            "timestamp": time.time(),
            "from_cache": False
        }
    
    def _apply_risk_management(self, analysis: Dict) -> Dict:
        """리스크 관리 적용"""
        try:
            risk_level = analysis.get("risk_level", "MEDIUM")
            final_confidence = analysis["final_confidence"]
            
            risk_multipliers = {
                "LOW": 1.2,
                "MEDIUM_LOW": 1.0,
                "MEDIUM": 0.8,
                "MEDIUM_HIGH": 0.6,
                "HIGH": 0.4,
                "VERY_HIGH": 0.2
            }
            
            multiplier = risk_multipliers.get(risk_level, 0.8)
            
            current_position = analysis.get("recommended_position_size", 0.1)
            analysis["recommended_position_size"] = min(
                current_position * multiplier,
                TRADING_CONFIG["max_position_size"]
            )
            
            if risk_level in ["HIGH", "VERY_HIGH"]:
                current_stop = analysis.get("stop_loss", 0.97)
                if current_stop < 1.0:
                    analysis["stop_loss"] = current_stop + (1.0 - current_stop) * 0.3
                else:
                    analysis["stop_loss"] = current_stop - (current_stop - 1.0) * 0.3
            
            analysis["risk_management_applied"] = True
            analysis["risk_multiplier"] = multiplier
            
        except Exception as e:
            logging.warning(f"리스크 관리 적용 실패: {e}")
        
        return analysis
    
    def _apply_dynamic_sizing(self, analysis: Dict, validation_result: Dict) -> Dict:
        """동적 포지션 사이징 적용"""
        try:
            base_size = analysis.get("recommended_position_size", 0.1)
            final_confidence = analysis["final_confidence"]
            
            market_analysis = validation_result.get("enhancements", {}).get("market_analysis", {})
            volume_score = market_analysis.get("volume_analysis", {}).get("score", 0.5)
            volatility_score = market_analysis.get("volatility_analysis", {}).get("score", 0.5)
            
            volume_multiplier = max(0.5, min(1.5, volume_score * 2))
            volatility_multiplier = max(0.3, min(1.2, volatility_score))
            confidence_multiplier = max(0.2, min(2.0, final_confidence * 1.5))
            
            dynamic_size = base_size * volume_multiplier * volatility_multiplier * confidence_multiplier
            analysis["recommended_position_size"] = min(dynamic_size, TRADING_CONFIG["max_position_size"])
            
            analysis["dynamic_sizing_applied"] = True
            analysis["sizing_factors"] = {
                "base_size": base_size,
                "volume_multiplier": volume_multiplier,
                "volatility_multiplier": volatility_multiplier,
                "confidence_multiplier": confidence_multiplier,
                "final_size": analysis["recommended_position_size"]
            }
            
        except Exception as e:
            logging.warning(f"동적 사이징 적용 실패: {e}")
        
        return analysis
    
    def _apply_kelly_formula_complete(self, analysis: Dict) -> Dict:
        """완전한 켈리 공식 적용"""
        try:
            final_confidence = analysis["final_confidence"]
            win_rate = final_confidence
            
            avg_win = 1.03
            avg_loss = 0.97
            
            if "stop_loss" in analysis and "take_profit" in analysis:
                avg_win = analysis["take_profit"]
                avg_loss = analysis["stop_loss"]
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * abs(1 - avg_loss)) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, TRADING_CONFIG["kelly_fraction"]))
            
            current_size = analysis.get("recommended_position_size", 0.1)
            kelly_adjusted_size = min(current_size, kelly_fraction)
            
            analysis["kelly_formula_applied"] = True
            analysis["kelly_calculation"] = {
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "kelly_fraction": kelly_fraction,
                "original_size": current_size,
                "kelly_adjusted_size": kelly_adjusted_size
            }
            
            if kelly_adjusted_size < current_size:
                analysis["recommended_position_size"] = kelly_adjusted_size
                analysis["kelly_reduction_applied"] = True
            
        except Exception as e:
            logging.warning(f"켈리 공식 적용 실패: {e}")
        
        return analysis
    
    def _apply_performance_adjustment_complete(self, analysis: Dict) -> Dict:
        """완전한 성과 기반 조정"""
        try:
            if len(self.performance_history) < 10:
                analysis["performance_adjustment"] = "INSUFFICIENT_DATA"
                return analysis
            
            recent_performances = list(self.performance_history)[-50:]
            
            successful_analyses = sum(1 for p in recent_performances if p["final_confidence"] > 0.7)
            total_analyses = len(recent_performances)
            success_rate = successful_analyses / total_analyses if total_analyses > 0 else 0.5
            
            avg_processing_time = sum(p["processing_time"] for p in recent_performances) / len(recent_performances)
            
            performance_multiplier = max(0.5, min(1.5, success_rate * 2))
            
            if avg_processing_time > 5.0:
                performance_multiplier *= 0.9
            
            current_confidence = analysis["final_confidence"]
            adjusted_confidence = current_confidence * performance_multiplier * TRADING_CONFIG["performance_adjustment"]
            analysis["final_confidence"] = min(1.0, adjusted_confidence)
            
            current_size = analysis.get("recommended_position_size", 0.1)
            adjusted_size = current_size * performance_multiplier
            analysis["recommended_position_size"] = min(adjusted_size, TRADING_CONFIG["max_position_size"])
            
            analysis["performance_adjustment_applied"] = True
            analysis["performance_stats"] = {
                "success_rate": success_rate,
                "total_analyses": total_analyses,
                "avg_processing_time": avg_processing_time,
                "performance_multiplier": performance_multiplier,
                "original_confidence": current_confidence,
                "adjusted_confidence": analysis["final_confidence"]
            }
            
        except Exception as e:
            logging.warning(f"성과 기반 조정 실패: {e}")
        
        return analysis
    
    def _final_analysis_validation(self, analysis: Dict, validation_result: Dict) -> Dict:
        """최종 분석 검증"""
        try:
            # 최소 임계값 체크
            if analysis["final_confidence"] < TRADING_CONFIG["min_confidence"]:
                analysis["execution_blocked"] = True
                analysis["block_reason"] = "신뢰도 기준 미달"
            
            # 포지션 크기 검증
            if analysis["recommended_position_size"] > TRADING_CONFIG["max_position_size"]:
                analysis["recommended_position_size"] = TRADING_CONFIG["max_position_size"]
                analysis["position_size_capped"] = True
            
            # 리스크 레벨 최종 검증
            risk_assessment = validation_result.get("enhancements", {}).get("risk_assessment", {})
            if risk_assessment.get("risk_level") in ["HIGH", "VERY_HIGH"]:
                analysis["recommended_position_size"] *= 0.5
                analysis["high_risk_adjustment"] = True
            
            # 최종 등급 부여
            final_confidence = analysis["final_confidence"]
            if final_confidence >= 0.9:
                analysis["analysis_grade"] = "EXCEPTIONAL"
            elif final_confidence >= 0.8:
                analysis["analysis_grade"] = "EXCELLENT"
            elif final_confidence >= 0.7:
                analysis["analysis_grade"] = "GOOD"
            elif final_confidence >= 0.6:
                analysis["analysis_grade"] = "FAIR"
            elif final_confidence >= 0.5:
                analysis["analysis_grade"] = "POOR"
            else:
                analysis["analysis_grade"] = "UNACCEPTABLE"
            
            analysis["final_validation_completed"] = True
            
        except Exception as e:
            logging.warning(f"최종 분석 검증 실패: {e}")
            analysis["validation_error"] = str(e)
        
        return analysis
    
    def _determine_execution_timing(self, confidence: float, risk_score: float) -> str:
        """실행 타이밍 결정"""
        if confidence > 0.85 and risk_score < 0.3:
            return "IMMEDIATE"
        elif confidence > 0.75 and risk_score < 0.5:
            return "URGENT"
        elif confidence > 0.65 and risk_score < 0.7:
            return "CAREFUL"
        elif confidence > 0.5:
            return "DELAYED"
        else:
            return "HOLD"
    
    def _emergency_complete_analysis(self, signal_data: Dict, error_msg: str) -> Dict:
        """응급 완전 분석"""
        return {
            "analysis_type": "EMERGENCY_COMPLETE",
            "analysis_version": "4.0.0-emergency",
            "error": error_msg,
            "original_confidence": signal_data.get("confidence", 0.8),
            "final_confidence": 0.3,
            "recommended_position_size": 0.02,
            "risk_level": "VERY_HIGH",
            "execution_timing": "HOLD",
            "urgency": 1,
            "emergency_mode": True,
            "timestamp": time.time()
        }
    
    def _update_analysis_stats(self, processing_time: float):
        """분석 통계 업데이트"""
        total = self.analysis_stats["total_analyses"]
        current_avg = self.analysis_stats["avg_processing_time"]
        
        # 이동 평균으로 처리 시간 업데이트
        self.analysis_stats["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        ) if total > 0 else processing_time

# 완전한 거래 실행기
class CompleteTradeExecutor:
    def __init__(self):
        self.execution_history = deque(maxlen=1000)
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0,
            "total_profit_loss": 0.0
        }
        self.active_positions = {}
        self.risk_limits = {
            "max_daily_trades": 50,
            "max_concurrent_positions": 10,
            "max_daily_loss": 0.05,  # 5%
            "max_position_correlation": 0.7
        }
        
    async def execute_trade_complete(self, signal_data: Dict, analysis_result: Dict) -> Dict:
        """완전한 거래 실행"""
        execution_id = self._generate_execution_id()
        start_time = time.time()
        
        try:
            # 실행 전 검증
            pre_execution_check = self._pre_execution_validation(signal_data, analysis_result)
            if not pre_execution_check["can_execute"]:
                return self._create_execution_result(
                    execution_id, "BLOCKED", pre_execution_check["reason"],
                    signal_data, analysis_result, start_time
                )
            
            # 리스크 한도 체크
            risk_check = self._check_risk_limits(signal_data, analysis_result)
            if not risk_check["within_limits"]:
                return self._create_execution_result(
                    execution_id, "RISK_BLOCKED", risk_check["reason"],
                    signal_data, analysis_result, start_time
                )
            
            # 포지션 상관관계 체크
            correlation_check = self._check_position_correlation(signal_data)
            if not correlation_check["acceptable"]:
                return self._create_execution_result(
                    execution_id, "CORRELATION_BLOCKED", correlation_check["reason"],
                    signal_data, analysis_result, start_time
                )
            
            # 실제 거래 실행 (시뮬레이션)
            execution_result = await self._execute_trade_simulation(
                execution_id, signal_data, analysis_result
            )
            
            # 포지션 추적 시작
            if execution_result["status"] == "EXECUTED":
                self._start_position_tracking(execution_id, signal_data, analysis_result, execution_result)
            
            # 실행 기록 저장
            self._record_execution(execution_result)
            
            return execution_result
            
        except Exception as e:
            logging.error(f"거래 실행 중 오류: {e}\n{traceback.format_exc()}")
            return self._create_execution_result(
                execution_id, "ERROR", str(e),
                signal_data, analysis_result, start_time
            )
    
    def _generate_execution_id(self) -> str:
        """실행 ID 생성"""
        timestamp = int(time.time() * 1000)
        return f"EXEC_{timestamp}_{np.random.randint(1000, 9999)}"
    
    def _pre_execution_validation(self, signal_data: Dict, analysis_result: Dict) -> Dict:
        """실행 전 검증"""
        can_execute = True
        reasons = []
        
        # 분석 결과 검증
        if analysis_result.get("execution_blocked", False):
            can_execute = False
            reasons.append(analysis_result.get("block_reason", "분석 결과 차단"))
        
        # 신뢰도 검증
        if analysis_result["final_confidence"] < TRADING_CONFIG["min_confidence"]:
            can_execute = False
            reasons.append(f"신뢰도 부족: {analysis_result['final_confidence']:.3f}")
        
        # 실행 타이밍 검증
        execution_timing = analysis_result.get("execution_timing", "HOLD")
        if execution_timing == "HOLD":
            can_execute = False
            reasons.append("실행 대기 상태")
        
        # 시장 조건 검증
        validation_data = analysis_result.get("validation_data", {})
        market_analysis = validation_data.get("enhancements", {}).get("market_analysis", {})
        if not market_analysis.get("trading_favorable", True):
            can_execute = False
            reasons.append("불리한 시장 조건")
        
        return {
            "can_execute": can_execute,
            "reason": "; ".join(reasons) if reasons else "검증 통과"
        }
    
    def _check_risk_limits(self, signal_data: Dict, analysis_result: Dict) -> Dict:
        """리스크 한도 체크"""
        within_limits = True
        reasons = []
        
        # 일일 거래 횟수 체크
        today = datetime.now().date()
        today_executions = sum(1 for exec_data in self.execution_history 
                              if datetime.fromtimestamp(exec_data["timestamp"]).date() == today)
        
        if today_executions >= self.risk_limits["max_daily_trades"]:
            within_limits = False
            reasons.append(f"일일 거래 한도 초과: {today_executions}")
        
        # 동시 포지션 수 체크
        active_count = len(self.active_positions)
        if active_count >= self.risk_limits["max_concurrent_positions"]:
            within_limits = False
            reasons.append(f"동시 포지션 한도 초과: {active_count}")
        
        # 일일 손실 한도 체크
        today_pnl = sum(exec_data.get("simulated_pnl", 0) for exec_data in self.execution_history 
                       if datetime.fromtimestamp(exec_data["timestamp"]).date() == today)
        
        if today_pnl < -self.risk_limits["max_daily_loss"]:
            within_limits = False
            reasons.append(f"일일 손실 한도 초과: {today_pnl:.3f}")
        
        # 포지션 크기 검증
        position_size = analysis_result.get("recommended_position_size", 0)
        if position_size > TRADING_CONFIG["max_position_size"]:
            within_limits = False
            reasons.append(f"포지션 크기 초과: {position_size:.3f}")
        
        return {
            "within_limits": within_limits,
            "reason": "; ".join(reasons) if reasons else "한도 내"
        }
    
    def _check_position_correlation(self, signal_data: Dict) -> Dict:
        """포지션 상관관계 체크"""
        symbol = signal_data["symbol"]
        action = signal_data["action"].lower()
        
        # 동일 심볼 포지션 체크
        same_symbol_positions = [pos for pos in self.active_positions.values() 
                               if pos["symbol"] == symbol]
        
        if same_symbol_positions:
            # 같은 방향 포지션이 이미 있는지 체크
            same_direction = any(pos["action"].lower() == action for pos in same_symbol_positions)
            if same_direction:
                return {
                    "acceptable": False,
                    "reason": f"동일 심볼 {symbol}에 같은 방향 포지션 존재"
                }
        
        # 상관관계가 높은 심볼들 체크
        correlated_symbols = self._get_correlated_symbols(symbol)
        correlated_positions = [pos for pos in self.active_positions.values() 
                              if pos["symbol"] in correlated_symbols and pos["action"].lower() == action]
        
        if len(correlated_positions) >= 3:  # 3개 이상의 상관관계 높은 포지션
            return {
                "acceptable": False,
                "reason": f"상관관계 높은 포지션 과다: {len(correlated_positions)}개"
            }
        
        return {"acceptable": True, "reason": "상관관계 체크 통과"}
    
    def _get_correlated_symbols(self, symbol: str) -> List[str]:
        """상관관계 높은 심볼들 반환"""
        correlation_groups = {
            "BTCUSDT": ["ETHUSDT", "ADAUSDT", "DOGEUSDT"],
            "ETHUSDT": ["BTCUSDT", "ADAUSDT", "LINKUSDT"],
            "BNBUSDT": ["ADAUSDT", "DOGEUSDT"],
        }
        return correlation_groups.get(symbol, [])
    
    async def _execute_trade_simulation(self, execution_id: str, signal_data: Dict, analysis_result: Dict) -> Dict:
        """거래 실행 시뮬레이션 (레버리지 포함)"""
        try:
            # 시뮬레이션 실행 지연 (실제 거래소 API 호출 시뮬레이션)
            await asyncio.sleep(np.random.uniform(0.1, 0.5))
            
            symbol = signal_data["symbol"]
            action = signal_data["action"]
            price = float(signal_data["price"])
            
            # 레버리지 정보 가져오기
            leverage_analysis = analysis_result.get("leverage_analysis", {})
            leverage = leverage_analysis.get("leverage", 1)
            actual_position_size = leverage_analysis.get("actual_position_size", 100)
            margin_required = leverage_analysis.get("margin_required", 100)
            liquidation_price = leverage_analysis.get("liquidation_price", 0)
            
            # 실행 가격 계산 (슬리피지 시뮬레이션)
            slippage = np.random.uniform(0.0001, 0.002)  # 0.01% ~ 0.2% 슬리피지
            if action.lower() in ["buy", "long"]:
                execution_price = price * (1 + slippage)
            else:
                execution_price = price * (1 - slippage)
            
            # 실행 성공률 시뮬레이션 (분석 품질에 따라)
            success_probability = min(0.95, analysis_result["final_confidence"] * 1.2)
            execution_successful = np.random.random() < success_probability
            
            if execution_successful:
                # 레버리지 거래 수수료 계산
                trading_fee_rate = LEVERAGE_CONFIG["trading_fee"]
                trading_fee = actual_position_size * trading_fee_rate
                
                # 펀딩 수수료 (8시간마다)
                funding_fee_rate = LEVERAGE_CONFIG["funding_fee"]
                funding_fee = actual_position_size * funding_fee_rate
                
                total_fees = trading_fee + funding_fee
                
                # 예상 P&L 계산 (레버리지 적용)
                # 2% 움직임 시 실제 수익은 leverage * 2%
                expected_return_pct = np.random.uniform(-0.02, 0.04)  # -2% ~ +4%
                leveraged_return_pct = expected_return_pct * leverage
                
                # 실제 P&L (마진 기준)
                pnl_on_margin = margin_required * leveraged_return_pct
                net_pnl = pnl_on_margin - total_fees
                
                # ROE (Return on Equity) 계산
                roe_percent = (net_pnl / margin_required) * 100
                
                # 리스크 메트릭
                distance_to_liquidation = abs(execution_price - liquidation_price) / execution_price * 100
                
                return self._create_execution_result(
                    execution_id, "EXECUTED", "레버리지 거래 실행 성공",
                    signal_data, analysis_result, time.time(),
                    {
                        "execution_price": execution_price,
                        "slippage": slippage,
                        
                        # 레버리지 거래 정보
                        "leverage": leverage,
                        "margin_mode": "ISOLATED",
                        "actual_position_size": actual_position_size,
                        "margin_required": margin_required,
                        "liquidation_price": liquidation_price,
                        
                        # 수수료
                        "trading_fee": trading_fee,
                        "funding_fee": funding_fee,
                        "total_fees": total_fees,
                        
                        # P&L
                        "expected_return_pct": expected_return_pct,
                        "leveraged_return_pct": leveraged_return_pct,
                        "pnl_on_margin": pnl_on_margin,
                        "net_pnl": net_pnl,
                        "roe_percent": roe_percent,
                        
                        # 리스크 메트릭
                        "distance_to_liquidation_pct": distance_to_liquidation,
                        "risk_level": leverage_analysis.get("risk_level", "MEDIUM"),
                        
                        # 익절/손절 정보
                        "stop_loss_price": execution_price * analysis_result.get("stop_loss", 0.98),
                        "take_profit_price": execution_price * analysis_result.get("take_profit", 1.02),
                        "stop_loss_percent": analysis_result.get("stop_loss_percent", 2),
                        "take_profit_percent": analysis_result.get("take_profit_percent", 2)
                    }
                )
            else:
                return self._create_execution_result(
                    execution_id, "FAILED", "레버리지 거래 실행 실패 (시장 조건)",
                    signal_data, analysis_result, time.time()
                )
                
        except Exception as e:
            return self._create_execution_result(
                execution_id, "ERROR", f"레버리지 거래 실행 중 오류: {e}",
                signal_data, analysis_result, time.time()
            )
    
    def _create_execution_result(self, execution_id: str, status: str, message: str,
                               signal_data: Dict, analysis_result: Dict, start_time: float,
                               execution_details: Dict = None) -> Dict:
        """실행 결과 생성"""
        result = {
            "execution_id": execution_id,
            "status": status,
            "message": message,
            "timestamp": time.time(),
            "execution_time": time.time() - start_time,
            "signal_data": signal_data,
            "analysis_result": analysis_result,
            "execution_details": execution_details or {}
        }
        
        return result
    
    def _start_position_tracking(self, execution_id: str, signal_data: Dict, 
                               analysis_result: Dict, execution_result: Dict):
        """포지션 추적 시작 (레버리지 포함)"""
        execution_details = execution_result.get("execution_details", {})
        
        position_data = {
            "execution_id": execution_id,
            "symbol": signal_data["symbol"],
            "action": signal_data["action"],
            "entry_price": execution_details.get("execution_price", signal_data["price"]),
            "position_size": analysis_result["recommended_position_size"],
            
            # 레버리지 정보
            "leverage": execution_details.get("leverage", 1),
            "margin_mode": "ISOLATED",
            "actual_position_size": execution_details.get("actual_position_size", 100),
            "margin_required": execution_details.get("margin_required", 100),
            "liquidation_price": execution_details.get("liquidation_price", 0),
            
            # 익절/손절 (2% 고정)
            "stop_loss": analysis_result.get("stop_loss"),
            "take_profit": analysis_result.get("take_profit"),
            "stop_loss_price": execution_details.get("stop_loss_price"),
            "take_profit_price": execution_details.get("take_profit_price"),
            "stop_loss_percent": 2.0,
            "take_profit_percent": 2.0,
            
            "entry_time": time.time(),
            "status": "ACTIVE",
            
            # 수수료 정보
            "trading_fee": execution_details.get("trading_fee", 0),
            "funding_fee": execution_details.get("funding_fee", 0),
            
            # 리스크 정보
            "risk_level": execution_details.get("risk_level", "MEDIUM"),
            "distance_to_liquidation_pct": execution_details.get("distance_to_liquidation_pct", 50)
        }
        
        self.active_positions[execution_id] = position_data
        
        # 백그라운드에서 포지션 모니터링 시작
        asyncio.create_task(self._monitor_position(execution_id))
    
    async def _monitor_position(self, execution_id: str):
        """포지션 모니터링 (레버리지 포함)"""
        try:
            position = self.active_positions.get(execution_id)
            if not position:
                return
            
            symbol = position["symbol"]
            entry_price = position["entry_price"]
            stop_loss = position.get("stop_loss")
            take_profit = position.get("take_profit")
            action = position["action"].lower()
            leverage = position.get("leverage", 1)
            liquidation_price = position.get("liquidation_price", 0)
            actual_position_size = position.get("actual_position_size", 100)
            margin_required = position.get("margin_required", 100)
            
            while execution_id in self.active_positions:
                # 현재 가격 조회 (실제로는 API 호출)
                price_change = np.random.uniform(-0.05, 0.05)  # ±5% 변동
                current_price = entry_price * (1 + price_change)
                
                # 청산가 체크 (최우선)
                liquidation_triggered = False
                if liquidation_price > 0:
                    if action in ["buy", "long"] and current_price <= liquidation_price:
                        liquidation_triggered = True
                    elif action in ["sell", "short"] and current_price >= liquidation_price:
                        liquidation_triggered = True
                
                if liquidation_triggered:
                    await self._close_position(execution_id, liquidation_price, "강제 청산 (LIQUIDATION)")
                    break
                
                # 손절/익절 체크
                should_close = False
                close_reason = ""
                
                # 2% 손절 체크
                if action in ["buy", "long"] and stop_loss:
                    if current_price <= entry_price * stop_loss:  # 98% 이하
                        should_close = True
                        close_reason = "손절선 도달 (-2%)"
                elif action in ["sell", "short"] and stop_loss:
                    if current_price >= entry_price * stop_loss:  # 102% 이상
                        should_close = True
                        close_reason = "손절선 도달 (-2%)"
                
                # 2% 익절 체크
                if action in ["buy", "long"] and take_profit:
                    if current_price >= entry_price * take_profit:  # 102% 이상
                        should_close = True
                        close_reason = "익절선 도달 (+2%)"
                elif action in ["sell", "short"] and take_profit:
                    if current_price <= entry_price * take_profit:  # 98% 이하
                        should_close = True
                        close_reason = "익절선 도달 (+2%)"
                
                # 시간 기반 청산 (24시간 후 자동 청산)
                if time.time() - position["entry_time"] > 86400:
                    should_close = True
                    close_reason = "시간 만료 (24시간)"
                
                # 리스크 관리: 청산가에 너무 가까우면 조기 청산
                if liquidation_price > 0:
                    distance_to_liquidation = abs(current_price - liquidation_price) / current_price
                    if distance_to_liquidation < 0.05:  # 5% 이내로 접근
                        should_close = True
                        close_reason = "청산 위험 (5% 이내 접근)"
                
                if should_close:
                    await self._close_position(execution_id, current_price, close_reason)
                    break
                
                # 10초마다 체크
                await asyncio.sleep(10)
                
        except Exception as e:
            logging.error(f"포지션 모니터링 오류 ({execution_id}): {e}")
            if execution_id in self.active_positions:
                del self.active_positions[execution_id]
    
    async def _close_position(self, execution_id: str, close_price: float, reason: str):
        """포지션 청산 (레버리지 포함)"""
        try:
            position = self.active_positions.get(execution_id)
            if not position:
                return
            
            entry_price = position["entry_price"]
            position_size = position["position_size"]
            action = position["action"].lower()
            leverage = position.get("leverage", 1)
            actual_position_size = position.get("actual_position_size", 100)
            margin_required = position.get("margin_required", 100)
            trading_fee = position.get("trading_fee", 0)
            
            # 가격 변동률 계산
            if action in ["buy", "long"]:
                price_change_pct = (close_price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - close_price) / entry_price
            
            # 레버리지 적용 수익률
            leveraged_return_pct = price_change_pct * leverage
            
            # P&L 계산 (마진 기준)
            pnl_on_margin = margin_required * leveraged_return_pct
            
            # 청산 수수료 계산
            closing_fee = actual_position_size * LEVERAGE_CONFIG["trading_fee"]
            total_fees = trading_fee + closing_fee
            
            # 최종 순손익
            net_pnl = pnl_on_margin - total_fees
            
            # ROE 계산
            roe_percent = (net_pnl / margin_required) * 100
            
            # 포지션 업데이트
            position["status"] = "CLOSED"
            position["close_price"] = close_price
            position["close_time"] = time.time()
            position["close_reason"] = reason
            position["price_change_pct"] = price_change_pct * 100
            position["leveraged_return_pct"] = leveraged_return_pct * 100
            position["pnl_on_margin"] = pnl_on_margin
            position["closing_fee"] = closing_fee
            position["total_fees"] = total_fees
            position["net_pnl"] = net_pnl
            position["roe_percent"] = roe_percent
            
            # 통계 업데이트
            self.execution_stats["total_profit_loss"] += net_pnl
            
            # 액티브 포지션에서 제거
            del self.active_positions[execution_id]
            
            # 상세 로그
            logging.info(
                f"🔥 레버리지 포지션 청산: {execution_id}\n"
                f"   📊 심볼: {position['symbol']} ({leverage}x)\n"
                f"   📈 액션: {position['action'].upper()}\n"
                f"   💰 진입가: ${entry_price:,.2f}\n"
                f"   💰 청산가: ${close_price:,.2f}\n"
                f"   📊 가격변동: {price_change_pct*100:+.2f}%\n"
                f"   ⚡ 레버리지 수익률: {leveraged_return_pct*100:+.2f}%\n"
                f"   💵 실제 포지션: ${actual_position_size:,.2f}\n"
                f"   💸 마진: ${margin_required:,.2f}\n"
                f"   💰 순손익: ${net_pnl:+.2f}\n"
                f"   📊 ROE: {roe_percent:+.2f}%\n"
                f"   🔥 사유: {reason}"
            )
            
        except Exception as e:
            logging.error(f"포지션 청산 오류 ({execution_id}): {e}")
            traceback.print_exc()
    
    def _record_execution(self, execution_result: Dict):
        """실행 기록 저장"""
        self.execution_history.append(execution_result)
        self.execution_stats["total_executions"] += 1
        
        if execution_result["status"] == "EXECUTED":
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
        
        # 평균 실행 시간 업데이트
        total = self.execution_stats["total_executions"]
        current_avg = self.execution_stats["avg_execution_time"]
        new_time = execution_result["execution_time"]
        
        self.execution_stats["avg_execution_time"] = (
            (current_avg * (total - 1) + new_time) / total
        ) if total > 0 else new_time

# 완전한 성능 모니터링 시스템
class CompletePerformanceMonitor:
    def __init__(self):
        self.metrics_history = deque(maxlen=10000)
        self.alert_thresholds = {
            "memory_usage": 0.8,  # 80%
            "cpu_usage": 0.9,     # 90%
            "error_rate": 0.1,    # 10%
            "response_time": 5.0,  # 5초
            "queue_size": 1000
        }
        self.monitoring_active = True
        self.last_cleanup = time.time()
        
    async def start_monitoring(self):
        """성능 모니터링 시작"""
        logging.info("🔍 완전한 성능 모니터링 시작")
        
        while self.monitoring_active:
            try:
                # 시스템 메트릭 수집
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # 알림 체크
                await self._check_alerts(metrics)
                
                # 자동 정리 (10분마다)
                if time.time() - self.last_cleanup > 600:
                    await self._automatic_cleanup()
                    self.last_cleanup = time.time()
                
                # 30초마다 수집
                await asyncio.sleep(30)
                
            except Exception as e:
                logging.error(f"성능 모니터링 오류: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> Dict:
        """시스템 메트릭 수집"""
        try:
            # 메모리 사용량
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent / 100
            
            # CPU 사용량
            cpu_usage = psutil.cpu_percent(interval=1) / 100
            
            # 디스크 사용량
            disk_info = psutil.disk_usage('/')
            disk_usage = disk_info.percent / 100
            
            # 프로세스 정보
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = process.cpu_percent()
            
            # 네트워크 통계
            network_info = psutil.net_io_counters()
            
            return {
                "timestamp": time.time(),
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "disk_usage": disk_usage,
                "process_memory_mb": process_memory,
                "process_cpu_percent": process_cpu,
                "network_bytes_sent": network_info.bytes_sent,
                "network_bytes_recv": network_info.bytes_recv,
                "active_threads": threading.active_count()
            }
            
        except Exception as e:
            logging.warning(f"메트릭 수집 실패: {e}")
            return {"timestamp": time.time(), "error": str(e)}
    
    async def _check_alerts(self, metrics: Dict):
        """알림 체크"""
        alerts = []
        
        # 메모리 사용량 체크
        if metrics.get("memory_usage", 0) > self.alert_thresholds["memory_usage"]:
            alerts.append(f"높은 메모리 사용량: {metrics['memory_usage']*100:.1f}%")
        
        # CPU 사용량 체크
        if metrics.get("cpu_usage", 0) > self.alert_thresholds["cpu_usage"]:
            alerts.append(f"높은 CPU 사용량: {metrics['cpu_usage']*100:.1f}%")
        
        # 프로세스 메모리 체크
        if metrics.get("process_memory_mb", 0) > 1000:  # 1GB
            alerts.append(f"높은 프로세스 메모리: {metrics['process_memory_mb']:.1f}MB")
        
        # 활성 스레드 수 체크
        if metrics.get("active_threads", 0) > 50:
            alerts.append(f"많은 활성 스레드: {metrics['active_threads']}개")
        
        # 알림 로그
        for alert in alerts:
            logging.warning(f"⚠️ 성능 알림: {alert}")
    
    async def _automatic_cleanup(self):
        """자동 정리"""
        try:
            # 가비지 컬렉션
            collected = gc.collect()
            logging.info(f"🧹 자동 정리 완료: {collected}개 객체 수집")
            
            # 로그 파일 크기 체크 및 로테이션
            await self._check_log_rotation()
            
        except Exception as e:
            logging.warning(f"자동 정리 실패: {e}")
    
    async def _check_log_rotation(self):
        """로그 로테이션 체크"""
        try:
            log_dir = Path("logs_complete_webhook")
            if not log_dir.exists():
                return
            
            for log_file in log_dir.glob("*.log"):
                # 100MB 이상 시 백업
                if log_file.stat().st_size > 100 * 1024 * 1024:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_file = log_file.parent / f"{log_file.stem}_{timestamp}.log"
                    log_file.rename(backup_file)
                    logging.info(f"📝 로그 파일 백업: {backup_file.name}")
                    
        except Exception as e:
            logging.warning(f"로그 로테이션 실패: {e}")
    
    def get_performance_summary(self) -> Dict:
        """성능 요약 반환"""
        if not self.metrics_history:
            return {"error": "메트릭 데이터 없음"}
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        avg_memory = np.mean([m.get("memory_usage", 0) for m in recent_metrics])
        avg_cpu = np.mean([m.get("cpu_usage", 0) for m in recent_metrics])
        avg_process_memory = np.mean([m.get("process_memory_mb", 0) for m in recent_metrics])
        
        return {
            "metrics_count": len(self.metrics_history),
            "avg_memory_usage": round(avg_memory * 100, 1),
            "avg_cpu_usage": round(avg_cpu * 100, 1),
            "avg_process_memory_mb": round(avg_process_memory, 1),
            "current_threads": threading.active_count(),
            "monitoring_duration": time.time() - self.metrics_history[0]["timestamp"] if self.metrics_history else 0
        }

# 신호 데이터 모델
class SignalModel(BaseModel):
    symbol: str
    action: str
    price: float
    confidence: Optional[float] = 0.8
    strategy: Optional[str] = "unknown"
    timeframe: Optional[str] = "1h"
    rsi: Optional[float] = None
    macd: Optional[float] = None
    volume: Optional[float] = None
    timestamp: Optional[str] = None
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        return v.upper().strip()
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        action = v.lower().strip()
        if action not in ['buy', 'sell', 'hold', 'long', 'short']:
            raise ValueError('action must be buy, sell, hold, long, or short')
        return action
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('price must be positive')
        return v
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if v is not None:
            return max(0.0, min(1.0, v))
        return v

# 완전한 웹훅 서버 애플리케이션
class CompleteWebhookServer:
    def __init__(self):
        self.app = FastAPI(
            title="Phoenix 95 Complete Webhook Server",
            description="완성된 기관급 암호화폐 거래 신호 처리 시스템",
            version="4.0.0-complete"
        )
        
        # 미들웨어 설정
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 컴포넌트 초기화
        self.signal_validator = CompleteSignalValidator()
        self.signal_analyzer = Phoenix95CompleteAnalyzer()
        self.trade_executor = CompleteTradeExecutor()
        self.performance_monitor = CompletePerformanceMonitor()
        
        # 서버 상태
        self.server_stats = {
            "start_time": time.time(),
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "active_connections": 0
        }
        
        # 신호 큐
        self.signal_queue = asyncio.Queue(maxsize=COMPLETE_WEBHOOK_CONFIG["signal_queue_size"])
        
        # 라우트 설정
        self._setup_routes()
        
        # 백그라운드 태스크
        self.background_tasks = []
        
    def _setup_routes(self):
        """라우트 설정"""
        
        @self.app.get("/")
        async def root():
            return HTMLResponse(self._generate_dashboard_html())
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime": time.time() - self.server_stats["start_time"],
                "version": "4.0.0-complete",
                "phoenix_95_available": PHOENIX_95_AVAILABLE,
                "active_connections": self.server_stats["active_connections"],
                "queue_size": self.signal_queue.qsize()
            }
        
        @self.app.post("/webhook/signal")
        async def receive_signal(request: Request):
            try:
                body = await request.body()
                body_str = body.decode('utf-8')
                
                # {{rsi}} 같은 치환되지 않은 변수 제거
                import re
                cleaned_body = re.sub(r'\{\{[^}]+\}\}', '50', body_str)
                
                logging.info(f"🔍 원본: {body_str}")
                logging.info(f"✅ 정리된: {cleaned_body}")
                
                # JSON 파싱
                signal_data = json.loads(cleaned_body)
                
                # SignalModel 생성
                signal = SignalModel(**signal_data)
                
                # 처리
                self.server_stats["total_requests"] += 1
                
                # 신호 데이터 변환
                signal_data = signal.model_dump()
                signal_data["timestamp"] = signal_data.get("timestamp") or time.time()
                
                # 큐에 추가
                if self.signal_queue.full():
                    # 큐가 가득 차면 가장 오래된 신호 제거
                    try:
                        self.signal_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                
                await self.signal_queue.put(signal_data)
                
                # 백그라운드에서 처리
                asyncio.create_task(self._process_signal(signal_data))
                
                self.server_stats["successful_requests"] += 1
                
                return {
                    "status": "received",
                    "message": "신호가 성공적으로 수신되었습니다",
                    "signal_id": f"SIG_{int(time.time() * 1000)}",
                    "queue_position": self.signal_queue.qsize(),
                    "timestamp": time.time()
                }
                
            except Exception as e:
                self.server_stats["failed_requests"] += 1
                logging.error(f"신호 수신 오류: {e}")
                return {"status": "error", "message": str(e)}
        
        @self.app.get("/stats")
        async def get_stats():
            validation_stats = self.signal_validator.validation_stats
            analysis_stats = self.signal_analyzer.analysis_stats
            execution_stats = self.trade_executor.execution_stats
            performance_stats = self.performance_monitor.get_performance_summary()
            
            return {
                "server": self.server_stats,
                "validation": validation_stats,
                "analysis": analysis_stats,
                "execution": execution_stats,
                "performance": performance_stats,
                "active_positions": len(self.trade_executor.active_positions),
                "timestamp": time.time()
            }
        
        @self.app.get("/positions")
        async def get_positions():
            return {
                "active_positions": dict(self.trade_executor.active_positions),
                "position_count": len(self.trade_executor.active_positions),
                "timestamp": time.time()
            }
        
        @self.app.post("/admin/shutdown")
        async def shutdown_server():
            return {"status": "shutting_down"}
    
    async def _process_signal(self, signal_data: Dict):
        """신호 처리"""
        try:
            start_time = time.time()
            
            # 1. 신호 검증
            validation_result = await self.signal_validator.validate_signal_complete(signal_data)
            
            if not validation_result["valid"]:
                logging.warning(f"신호 검증 실패: {signal_data['symbol']} - {validation_result.get('errors', [])}")
                return
            
            # 2. 신호 분석
            analysis_result = await self.signal_analyzer.analyze_signal_phoenix_95_complete(
                signal_data, validation_result
            )
            
            # 3. 거래 실행 (조건 만족 시)
            if (analysis_result["final_confidence"] >= TRADING_CONFIG["min_confidence"] and
                analysis_result.get("execution_timing") not in ["HOLD", "DELAYED"]):
                
                execution_result = await self.trade_executor.execute_trade_complete(
                    signal_data, analysis_result
                )
                
                logging.info(
                    f"거래 실행 완료: {execution_result['execution_id']} "
                    f"상태: {execution_result['status']} "
                    f"메시지: {execution_result['message']}"
                )
                
                # 텔레그램 알림 전송
                try:
                    leverage_info = analysis_result.get("leverage_analysis", {})
                    leverage = leverage_info.get("leverage", 1)
                    actual_position = leverage_info.get("actual_position_size", 0)
                    margin_required = leverage_info.get("margin_required", 0)
                    
                    telegram_message = (
                        f"🚀 <b>Phoenix 95 레버리지 거래</b>\n"
                        f"📊 심볼: {signal_data['symbol']}\n"
                        f"📈 액션: {signal_data['action'].upper()}\n"
                        f"💰 가격: ${signal_data['price']:,.2f}\n"
                        f"⚡ 레버리지: {leverage}x (ISOLATED)\n"
                        f"💵 실제 포지션: ${actual_position:,.2f}\n"
                        f"💸 필요 마진: ${margin_required:,.2f}\n"
                        f"🎯 신뢰도: {analysis_result['final_confidence']:.1%}\n"
                        f"📊 익절: +2% | 손절: -2%\n"
                        f"⚡ 분석: {analysis_result['analysis_type']}\n"
                        f"📊 실행: {execution_result['status']}"
                    )
                    
                    await send_telegram_signal(telegram_message)
                    
                except Exception as telegram_error:
                    logging.warning(f"텔레그램 전송 실패: {telegram_error}")
            
            processing_time = time.time() - start_time
            logging.info(
                f"신호 처리 완료: {signal_data['symbol']} {signal_data['action']} "
                f"처리시간: {processing_time:.3f}s"
            )
            
        except Exception as e:
            logging.error(f"신호 처리 중 오류: {e}\n{traceback.format_exc()}")

    def _generate_dashboard_html(self) -> str:
        """대시보드 HTML 생성"""
        uptime = time.time() - self.server_stats["start_time"]        
        uptime_str = str(timedelta(seconds=int(uptime)))
        validation_stats = self.signal_validator.validation_stats
        analysis_stats = self.signal_analyzer.analysis_stats
        execution_stats = self.trade_executor.execution_stats
        
        success_rate = (
            validation_stats["valid_signals"] / validation_stats["total_validated"] * 100
            if validation_stats["total_validated"] > 0 else 0
        )
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phoenix 95 Complete Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .stat-card {{ background: #2d2d2d; border-radius: 10px; padding: 20px; border-left: 5px solid #00ff88; }}
                .stat-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #00ff88; }}
                .stat-item {{ display: flex; justify-content: space-between; margin: 8px 0; }}
                .stat-value {{ color: #00ff88; font-weight: bold; }}
                .status-indicator {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
                .status-healthy {{ background: #00ff88; }}
                .footer {{ text-align: center; margin-top: 30px; color: #888; }}
            </style>
            <script>
                setInterval(() => location.reload(), 30000);
            </script>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Phoenix 95 Complete Webhook Server</h1>
                <p><span class="status-indicator status-healthy"></span>서버 상태: 정상 운영중</p>
                <p>업타임: {uptime_str} | Phoenix 95: {"✅ 활성" if PHOENIX_95_AVAILABLE else "❌ 비활성"}</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">📊 서버 통계</div>
                    <div class="stat-item">
                        <span>총 요청 수:</span>
                        <span class="stat-value">{self.server_stats["total_requests"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>성공한 요청:</span>
                        <span class="stat-value">{self.server_stats["successful_requests"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>실패한 요청:</span>
                        <span class="stat-value">{self.server_stats["failed_requests"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>대기 중인 신호:</span>
                        <span class="stat-value">{self.signal_queue.qsize()}</span>
                    </div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">🔍 신호 검증</div>
                    <div class="stat-item">
                        <span>총 검증:</span>
                        <span class="stat-value">{validation_stats["total_validated"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>유효한 신호:</span>
                        <span class="stat-value">{validation_stats["valid_signals"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>성공률:</span>
                        <span class="stat-value">{success_rate:.1f}%</span>
                    </div>
                    <div class="stat-item">
                        <span>API 호출:</span>
                        <span class="stat-value">{validation_stats["api_calls"]:,}</span>
                    </div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">🧠 AI 분석</div>
                    <div class="stat-item">
                        <span>총 분석:</span>
                        <span class="stat-value">{analysis_stats["total_analyses"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>Phoenix 95 분석:</span>
                        <span class="stat-value">{analysis_stats["phoenix_95_analyses"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>강화 분석:</span>
                        <span class="stat-value">{analysis_stats["enhanced_analyses"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>평균 처리시간:</span>
                        <span class="stat-value">{analysis_stats["avg_processing_time"]:.3f}s</span>
                    </div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">💰 레버리지 거래</div>
                    <div class="stat-item">
                        <span>총 실행:</span>
                        <span class="stat-value">{execution_stats["total_executions"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>성공한 거래:</span>
                        <span class="stat-value">{execution_stats["successful_executions"]:,}</span>
                    </div>
                    <div class="stat-item">
                        <span>활성 포지션:</span>
                        <span class="stat-value">{len(self.trade_executor.active_positions)}</span>
                    </div>
                    <div class="stat-item">
                        <span>총 P&L:</span>
                        <span class="stat-value">${execution_stats["total_profit_loss"]:.2f}</span>
                    </div>
                    <div class="stat-item">
                        <span>레버리지:</span>
                        <span class="stat-value">20x ISOLATED</span>
                    </div>
                    <div class="stat-item">
                        <span>익절/손절:</span>
                        <span class="stat-value">±2%</span>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Phoenix 95 Complete Webhook Server v4.0.0-LEVERAGE | 20x 이솔레이티드 레버리지 거래 시스템</p>
                <p>⚡ 레버리지: 20x | 📊 마진모드: ISOLATED | 🎯 익절/손절: ±2%</p>
                <p>마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    async def start_background_tasks(self):
        """백그라운드 태스크 시작"""
        logging.info("🔄 백그라운드 태스크 시작")
        
        # 성능 모니터링 시작
        monitor_task = asyncio.create_task(self.performance_monitor.start_monitoring())
        self.background_tasks.append(monitor_task)
        
        # 정기 정리 작업
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.background_tasks.append(cleanup_task)
        
        logging.info(f"✅ {len(self.background_tasks)}개 백그라운드 태스크 시작됨")
    
    async def _periodic_cleanup(self):
        """정기 정리 작업"""
        while True:
            try:
                await asyncio.sleep(COMPLETE_WEBHOOK_CONFIG["memory_cleanup_interval"])
                
                # 메모리 정리
                collected = gc.collect()
                logging.info(f"🧹 정기 정리: {collected}개 객체 수집됨")
                
                # 캐시 정리
                current_time = time.time()
                
                # 검증기 캐시 정리
                expired_keys = [
                    key for key, (_, timestamp) in self.signal_validator.price_cache.items()
                    if current_time - timestamp > self.signal_validator.cache_duration
                ]
                for key in expired_keys:
                    del self.signal_validator.price_cache[key]
                
                # 분석기 캐시 정리
                expired_keys = [
                    key for key, (_, timestamp) in self.signal_analyzer.analysis_cache.items()
                    if current_time - timestamp > self.signal_analyzer.cache_duration
                ]
                for key in expired_keys:
                    del self.signal_analyzer.analysis_cache[key]
                
                logging.info(f"🧹 캐시 정리 완료: 만료된 {len(expired_keys)}개 항목 제거")
                
            except Exception as e:
                logging.error(f"정기 정리 작업 오류: {e}")
    
    async def _graceful_shutdown(self):
        """안전한 종료"""
        logging.info("🛑 안전한 서버 종료 시작")
        
        # 성능 모니터링 중지
        self.performance_monitor.monitoring_active = False
        
        # 백그라운드 태스크 중지
        for task in self.background_tasks:
            task.cancel()
        
        # 활성 포지션 정리
        for execution_id in list(self.trade_executor.active_positions.keys()):
            await self.trade_executor._close_position(
                execution_id, 
                self.trade_executor.active_positions[execution_id]["entry_price"],
                "서버 종료"
            )
        
        logging.info("✅ 안전한 서버 종료 완료")

# 메인 실행 함수
async def main():
    """메인 실행 함수"""
    try:
        # 로깅 설정
        setup_complete_logging()
        
        # 서버 초기화
        webhook_server = CompleteWebhookServer()
        
        # ngrok 터널 시작 (활성화된 경우)
        ngrok_url = None
        if NGROK_CONFIG["enabled"]:
            ngrok_url = await start_ngrok_tunnel(COMPLETE_WEBHOOK_CONFIG["port"])
        
        # 백그라운드 태스크 시작
        await webhook_server.start_background_tasks()
        
        # 시그널 핸들러 설정
        def signal_handler(signum, frame):
            logging.info(f"종료 신호 수신: {signum}")
            asyncio.create_task(webhook_server._graceful_shutdown())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 서버 시작 메시지
        logging.info("🚀 Phoenix 95 Complete Webhook Server 시작")
        logging.info(f"📡 로컬 주소: http://{COMPLETE_WEBHOOK_CONFIG['host']}:{COMPLETE_WEBHOOK_CONFIG['port']}")
        if ngrok_url:
            logging.info(f"🌐 외부 접속 URL: {ngrok_url}")
            logging.info(f"📡 TradingView 웹훅: {ngrok_url}/webhook/signal")
        logging.info(f"🔧 Phoenix 95 엔진: {'✅ 활성' if PHOENIX_95_AVAILABLE else '❌ 비활성'}")
        logging.info(f"⚙️ 설정: {COMPLETE_WEBHOOK_CONFIG['environment']}")
        
        # 서버 실행
        config = uvicorn.Config(
            webhook_server.app,
            host=COMPLETE_WEBHOOK_CONFIG["host"],
            port=COMPLETE_WEBHOOK_CONFIG["port"],
            log_level=COMPLETE_WEBHOOK_CONFIG["log_level"],
            access_log=COMPLETE_WEBHOOK_CONFIG["access_log"],
            reload=COMPLETE_WEBHOOK_CONFIG["reload"]
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        logging.info("🛑 사용자에 의한 서버 종료")
    except Exception as e:
        logging.error(f"❌ 서버 실행 중 오류: {e}\n{traceback.format_exc()}")
    finally:
        logging.info("👋 Phoenix 95 Complete Webhook Server 종료")

if __name__ == "__main__":
    asyncio.run(main())