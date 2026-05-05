"use client";

import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

type Point = { lat: number; lon: number; name: string; type: string; condition: string };

const COLOR_BY_COND: Record<string, string> = {
  boa: "#10b981", regular: "#f59e0b", ruim: "#ef4444",
};

function pinIcon(color: string) {
  return L.divIcon({
    className: "",
    html: `<div style="width:18px;height:18px;border-radius:50%;background:${color};border:2px solid #0a0e1a;box-shadow:0 0 0 2px ${color}80"></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  });
}

export default function Map({ points }: { points: Point[] }) {
  useEffect(() => {
    // fix default icon paths after dynamic import
    delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: unknown })._getIconUrl;
  }, []);

  if (points.length === 0) return null;

  const center: [number, number] = [
    points.reduce((s, p) => s + p.lat, 0) / points.length,
    points.reduce((s, p) => s + p.lon, 0) / points.length,
  ];

  return (
    <MapContainer
      center={center}
      zoom={16}
      style={{ height: "100%", width: "100%", background: "#0a0e1a" }}
      scrollWheelZoom
    >
      <TileLayer
        attribution='&copy; OpenStreetMap'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {points.map((p, i) => (
        <Marker key={i} position={[p.lat, p.lon]} icon={pinIcon(COLOR_BY_COND[p.condition] || "#64748b")}>
          <Popup>
            <strong>{p.name}</strong><br />
            {p.type} · {p.condition}<br />
            <small>{p.lat.toFixed(5)}, {p.lon.toFixed(5)}</small>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
