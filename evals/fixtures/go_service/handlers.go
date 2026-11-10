package main

import "net/http"

func getHandler(c *Cache) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		value, ok := c.Get(r.URL.Query().Get("key"))
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte(value))
	}
}

func setHandler(c *Cache) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		c.Set(r.URL.Query().Get("key"), r.URL.Query().Get("value"))
		w.WriteHeader(http.StatusNoContent)
	}
}
