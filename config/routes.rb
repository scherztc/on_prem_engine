Rails.application.routes.draw do
  get 'chat/index'
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Defines the root path route ("/")
  # root "articles#index"

  root 'chat#index'
  post 'chat/send_message', to: 'chat#send_message'
end
