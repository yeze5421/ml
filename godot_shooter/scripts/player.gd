extends Area2D
class_name Player

signal health_changed(value: int)
signal died
signal request_shot(pos: Vector2)

@export var speed: float = 320.0
@export var max_health: int = 100
@export var shoot_cooldown: float = 0.2

var health: int
var _cooldown_left := 0.0
var _play_rect := Rect2(Vector2.ZERO, Vector2(960, 540))

func _ready() -> void:
	health = max_health
	health_changed.emit(health)

func setup_play_rect(rect: Rect2) -> void:
	_play_rect = rect

func _process(delta: float) -> void:
	var input_axis := Vector2(
		Input.get_action_strength("move_right") - Input.get_action_strength("move_left"),
		Input.get_action_strength("move_down") - Input.get_action_strength("move_up")
	)
	if input_axis.length() > 1.0:
		input_axis = input_axis.normalized()
	global_position += input_axis * speed * delta
	global_position.x = clamp(global_position.x, _play_rect.position.x + 24.0, _play_rect.end.x - 24.0)
	global_position.y = clamp(global_position.y, _play_rect.position.y + 24.0, _play_rect.end.y - 24.0)

	_cooldown_left = maxf(0.0, _cooldown_left - delta)
	if Input.is_action_pressed("shoot") and _cooldown_left <= 0.0:
		_cooldown_left = shoot_cooldown
		request_shot.emit(global_position + Vector2(0, -24))

func take_damage(amount: int) -> void:
	health = max(0, health - amount)
	health_changed.emit(health)
	if health <= 0:
		died.emit()
		queue_free()
