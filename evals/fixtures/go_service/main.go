package main

import (
	"log"
	"net/http"
)

func main() {
	store := NewCache()
	mux := http.NewServeMux()
	mux.HandleFunc("/get", getHandler(store))
	mux.HandleFunc("/set", setHandler(store))
	log.Fatal(http.ListenAndServe(":8080", mux))
}
