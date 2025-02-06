import requests
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import logging
from config import *
import json
import time

class NewsAnalyzer:
    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache = {}
        self.cache_duration = 3600  # 1 heure
        
    def get_news_for_symbol(self, symbol):
        """Récupère et analyse les news pour un symbole spécifique avec filtrage avancé"""
        try:
            # Vérifier le cache avec une clé plus précise
            cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d_%H_%M')}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                # Vérifier si le cache est encore valide (moins de 15 minutes)
                if (datetime.now() - datetime.strptime(cached_data.get('timestamp', ''), '%Y-%m-%d_%H_%M')).total_seconds() < 900:
                    return cached_data['data']
            
            if symbol not in SYMBOL_NEWS_KEYWORDS:
                logging.warning(f"Pas de mots-clés définis pour {symbol}")
                return self._get_empty_news_result()
            
            # Préparer la requête avec des paramètres optimisés
            keywords = self._build_advanced_query(symbol)
            from_date = (datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
            
            params = {
                'q': keywords,
                'from': from_date,
                'language': ",".join(NEWS_LANGUAGES),
                'sortBy': 'publishedAt',  # Priorité aux nouvelles les plus récentes
                'pageSize': 100,  # Augmenter le nombre d'articles
                'apiKey': self.api_key
            }
            
            # Faire la requête avec retry
            for attempt in range(3):
                try:
                    response = requests.get(self.base_url, params=params, timeout=10)
                    response.raise_for_status()
                    news_data = response.json()
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(1)
            
            if news_data['status'] != 'ok':
                raise Exception(f"Erreur API: {news_data.get('message', 'Unknown error')}")
            
            # Analyser les articles avec filtrage avancé
            articles = self._filter_relevant_articles(news_data['articles'], symbol)
            analyzed_articles = []
            total_sentiment = 0
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            weighted_impact = 0
            
            for article in articles:
                # Analyse avancée du sentiment
                title_sentiment = self._analyze_sentiment(article['title']) * 1.5  # Plus de poids au titre
                description_sentiment = self._analyze_sentiment(article['description'] or "") * 0.5
                sentiment_score = (title_sentiment + description_sentiment) / 2
                
                # Calcul d'impact amélioré
                impact = self._calculate_advanced_impact(article, symbol)
                weighted_impact += impact
                
                # Classification plus nuancée du sentiment
                sentiment_category = self._classify_sentiment(sentiment_score)
                sentiment_counts[sentiment_category] += 1
                
                analyzed_article = {
                    'title': article['title'],
                    'published_at': article['publishedAt'],
                    'url': article['url'],
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'sentiment_score': sentiment_score,
                    'sentiment_category': sentiment_category,
                    'impact_score': impact,
                    'relevance_score': self._calculate_relevance(article, symbol)
                }
                analyzed_articles.append(analyzed_article)
                total_sentiment += sentiment_score * impact  # Sentiment pondéré par l'impact
            
            # Calcul des scores finaux
            num_articles = len(analyzed_articles)
            if num_articles > 0:
                avg_sentiment = total_sentiment / sum(article['impact_score'] for article in analyzed_articles)
                impact_score = weighted_impact / num_articles
            else:
                avg_sentiment = 0
                impact_score = 0
            
            result = {
                "articles": sorted(analyzed_articles, key=lambda x: (x['impact_score'], x['published_at']), reverse=True),
                "sentiment_score": avg_sentiment,
                "summary": sentiment_counts,
                "impact_score": impact_score,
                "market_impact": self._calculate_market_impact(avg_sentiment, impact_score),
                "confidence_score": self._calculate_confidence_score(num_articles, sentiment_counts)
            }
            
            # Mise en cache avec timestamp
            self.cache[cache_key] = {
                'timestamp': datetime.now().strftime('%Y-%m-%d_%H_%M'),
                'data': result
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse des news pour {symbol}: {e}")
            return None
    
    def _analyze_sentiment(self, text):
        """Analyse avancée du sentiment d'un texte"""
        try:
            if not text:
                return 0
                
            # Analyse avec TextBlob
            analysis = TextBlob(text)
            
            # Facteurs de pondération
            weights = {
                'polarity': 0.7,
                'subjectivity': 0.3
            }
            
            # Calcul du score pondéré
            sentiment_score = (
                analysis.sentiment.polarity * weights['polarity'] +
                (1 - abs(analysis.sentiment.subjectivity - 0.5) * 2) * weights['subjectivity']
            )
            
            # Ajustement basé sur les mots-clés spécifiques au trading
            bullish_keywords = ['bullish', 'rally', 'surge', 'growth', 'recovery']
            bearish_keywords = ['bearish', 'decline', 'drop', 'fall', 'crisis']
            
            text_lower = text.lower()
            for keyword in bullish_keywords:
                if keyword in text_lower:
                    sentiment_score += 0.1
            for keyword in bearish_keywords:
                if keyword in text_lower:
                    sentiment_score -= 0.1
            
            return max(min(sentiment_score, 1.0), -1.0)  # Normalisation entre -1 et 1
            
        except Exception as e:
            logging.error(f"Erreur d'analyse de sentiment: {e}")
            return 0
    
    def _calculate_advanced_impact(self, article, symbol):
        """Calcul avancé du score d'impact d'un article"""
        impact_score = 0
        
        # Score basé sur la source avec pondération plus fine
        source_name = article.get('source', {}).get('name', '').lower()
        source_weights = {
            'reuters': 5,
            'bloomberg': 5,
            'financial times': 4,
            'wall street journal': 4,
            'cnbc': 3,
            'marketwatch': 3,
            'seeking alpha': 2
        }
        impact_score += source_weights.get(source_name, 1)
        
        # Score basé sur l'âge avec décroissance exponentielle
        pub_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
        age_hours = (datetime.now() - pub_date).total_seconds() / 3600
        time_factor = max(0, 1 - (age_hours / 24) ** 0.5)  # Décroissance plus rapide au début
        impact_score *= time_factor
        
        # Score basé sur la pertinence du contenu
        relevance_score = self._calculate_relevance(article, symbol)
        impact_score *= (0.5 + relevance_score)  # Augmentation jusqu'à 50% basée sur la pertinence
        
        # Bonus pour les articles contenant des données chiffrées
        if any(char.isdigit() for char in article['title'] + (article['description'] or "")):
            impact_score *= 1.2
        
        # Normalisation finale
        return min(impact_score / 10, 1.0)  # Score maximum de 1.0
    def _calculate_relevance(self, article, symbol):
        """Calcule la pertinence d'un article pour un symbole donné"""
        try:
            # Texte combiné pour l'analyse
            text = (article['title'] + " " + (article['description'] or "")).lower()
            
            # Mots-clés spécifiques au symbole
            keywords = SYMBOL_NEWS_KEYWORDS.get(symbol, [])
            
            # Calcul du score de pertinence
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
            base_score = keyword_matches / max(len(keywords), 1)
            
            # Bonus pour les mots-clés dans le titre
            title_matches = sum(1 for kw in keywords if kw.lower() in article['title'].lower())
            title_bonus = title_matches * 0.2
            
            # Pénalité pour les articles trop généraux
            general_terms = ['market', 'economy', 'stock', 'trading']
            generality_penalty = sum(1 for term in general_terms if term in text) * 0.1
            
            return min(base_score + title_bonus - generality_penalty, 1.0)
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul de la pertinence: {e}")
            return 0

    def _build_advanced_query(self, symbol):
        """Construit une requête avancée pour la recherche d'articles"""
        try:
            keywords = SYMBOL_NEWS_KEYWORDS.get(symbol, [])
            
            # Grouper les mots-clés par importance
            primary_keywords = [f'"{kw}"' for kw in keywords[:2]]  # Les plus importants
            secondary_keywords = [f'"{kw}"' for kw in keywords[2:]]
            
            # Construire la requête
            query_parts = []
            
            # Mots-clés principaux (au moins un requis)
            if primary_keywords:
                query_parts.append(f"({' OR '.join(primary_keywords)})")
            
            # Mots-clés secondaires (bonus)
            if secondary_keywords:
                query_parts.append(f"({' OR '.join(secondary_keywords)})")
            
            return " AND ".join(query_parts)
            
        except Exception as e:
            logging.error(f"Erreur lors de la construction de la requête: {e}")
            return " OR ".join(f'"{kw}"' for kw in SYMBOL_NEWS_KEYWORDS.get(symbol, []))

    def _filter_relevant_articles(self, articles, symbol):
        """Filtre les articles pour ne garder que les plus pertinents"""
        try:
            filtered_articles = []
            
            for article in articles:
                # Vérifier la présence de contenu
                if not article.get('title') or not article.get('description'):
                    continue
                
                # Calculer le score de pertinence
                relevance = self._calculate_relevance(article, symbol)
                
                # Filtrer les articles peu pertinents
                if relevance < 0.3:  # Seuil de pertinence minimum
                    continue
                
                # Ajouter le score de pertinence à l'article
                article['relevance_score'] = relevance
                filtered_articles.append(article)
            
            # Trier par pertinence et date
            return sorted(filtered_articles, 
                        key=lambda x: (x['relevance_score'], x['publishedAt']), 
                        reverse=True)
            
        except Exception as e:
            logging.error(f"Erreur lors du filtrage des articles: {e}")
            return articles

    def _classify_sentiment(self, score):
        """Classification plus nuancée du sentiment"""
        if score > NEWS_SENTIMENT_THRESHOLD:
            if score > 0.6:
                return "very_positive"
            return "positive"
        elif score < -NEWS_SENTIMENT_THRESHOLD:
            if score < -0.6:
                return "very_negative"
            return "negative"
        return "neutral"

    def _calculate_market_impact(self, sentiment_score, impact_score):
        """Calcule l'impact potentiel sur le marché"""
        combined_score = abs(sentiment_score * impact_score)
        
        if combined_score > 0.7:
            return "HIGH"
        elif combined_score > 0.3:
            return "MEDIUM"
        return "LOW"

    def _calculate_confidence_score(self, num_articles, sentiment_counts):
        """Calcule le score de confiance dans l'analyse"""
        try:
            if num_articles == 0:
                return 0
                
            # Calcul de la distribution des sentiments
            total = sum(sentiment_counts.values())
            distribution = {k: v/total for k, v in sentiment_counts.items()}
            
            # Calcul de la dominance du sentiment majoritaire
            max_sentiment = max(distribution.values())
            
            # Facteurs de confiance
            confidence_factors = {
                'num_articles': min(num_articles / 10, 1),  # Max 1.0 pour 10+ articles
                'sentiment_dominance': max_sentiment,
                'consistency': 1 - (distribution['neutral'] * 0.5)  # Pénalité pour sentiment neutre
            }
            
            # Score final
            confidence_score = (
                confidence_factors['num_articles'] * 0.4 +
                confidence_factors['sentiment_dominance'] * 0.4 +
                confidence_factors['consistency'] * 0.2
            )
            
            return min(confidence_score * 100, 100)  # Score sur 100
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul du score de confiance: {e}")
            return 50  # Score par défaut

    def _get_empty_news_result(self):
        """Retourne un résultat vide standardisé"""
        return {
            "articles": [],
            "sentiment_score": 0,
            "summary": {"positive": 0, "negative": 0, "neutral": 0},
            "impact_score": 0,
            "market_impact": "LOW",
            "confidence_score": 0
        }
        
    def get_market_sentiment(self, symbol):
        """Analyse le sentiment du marché pour un symbole donné"""
        try:
            news_data = self.get_news_for_symbol(symbol)
            if not news_data:
                return 0, 0  # Sentiment neutre et confiance nulle en cas d'erreur
                
            sentiment_score = news_data['sentiment_score']
            confidence_score = news_data['confidence_score']
            
            return sentiment_score, confidence_score
            
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse du sentiment de marché pour {symbol}: {e}")
            return 0, 0  # Sentiment neutre et confiance nulle en cas d'erreur
