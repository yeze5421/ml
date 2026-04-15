extends Node2D

@onready var timer: Timer = $Timer
@onready var particles: CPUParticles2D = $CPUParticles2D

func _ready() -> void:
	particles.restart()
	timer.start()

func _on_timer_timeout() -> void:
	queue_free()
