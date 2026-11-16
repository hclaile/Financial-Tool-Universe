"""Focused Reddit post and comment extraction."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def _record(post):
    post.comments.replace_more(limit=0)
    comments = []
    for comment in post.comments.list():
        body = " ".join(str(getattr(comment, "body", "")).split())
        if not body or body == "[deleted]":
            continue
        comments.append({
            "author": str(comment.author) if comment.author else "[deleted]",
            "score": comment.score,
            "text": body,
            "created_utc": datetime.fromtimestamp(comment.created_utc, timezone.utc).isoformat(),
        })
    return {
        "title": post.title,
        "author": str(post.author) if post.author else "[deleted]",
        "score": post.score,
        "url": post.url,
        "created_utc": datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat(),
        "subreddit": post.subreddit.display_name,
        "text": " ".join(str(post.selftext).split()),
        "top_comments": sorted(comments, key=lambda item: item["score"], reverse=True)[:10],
    }


def run(args):
    keyword = str(getattr(args, "keyword", "") or "").strip()
    output_path = str(getattr(args, "output_path", "") or "").strip()
    if not keyword or not output_path:
        raise ValueError("keyword and output_path are required")
    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    user_agent = os.environ.get("FTU_USER_AGENT", "")
    if not all((client_id, client_secret, user_agent)):
        raise ValueError("Reddit credentials or network.user_agent are missing in config.yaml")
    try:
        import praw
    except ImportError as exc:
        raise RuntimeError("crawler_reddit_extractor needs the crawlers extra (praw)") from exc

    start_raw = getattr(args, "start_time", None)
    end_raw = getattr(args, "end_time", None)
    start = date.fromisoformat(start_raw) if start_raw else date.today() - timedelta(days=30)
    end = date.fromisoformat(end_raw) if end_raw else date.today()
    if start > end:
        raise ValueError("start_time must be on or before end_time")
    limit = int(getattr(args, "limit", 100) or 100)
    if limit < 1:
        raise ValueError("limit must be positive")
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    time_filter = "all" if start < date.today() - timedelta(days=365) else "year"
    results = reddit.subreddit("all").search(query=keyword, sort="new", limit=limit, time_filter=time_filter)
    posts = []
    for post in results:
        created = datetime.fromtimestamp(post.created_utc, timezone.utc).date()
        if start <= created <= end:
            posts.append(_record(post))
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"keyword": keyword, "count": len(posts), "output_path": str(output), "posts": posts}
