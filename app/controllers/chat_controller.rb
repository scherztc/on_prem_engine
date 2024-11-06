require 'open3'

class ChatController < ApplicationController
  def index
  end

  def send_message
    user_message = params[:user_message]
    model_name = params[:model]  # Model name like "fastchat-t5-3b-v1.0"

    # Call the Python script to generate a response from the model
    response = chat_with_model(model_name, user_message)

    # Respond with Turbo Stream to update the page dynamically
    respond_to do |format|
      format.turbo_stream do
        render turbo_stream: turbo_stream.append("messages", partial: "message", locals: { message: response, user_message: user_message })
      end
    end  

#    render json: { response: response }
  end

  private

  def chat_with_model(model_name, message)
    # Call the Python script to interact with the Hugging Face model
    command = ["python3", "chat_with_model.py", model_name, message]
    
    # Capture the output of the Python script
    output, error, status = Open3.capture3(*command)
    
    if status.success?
      output.strip
    else
      "Error: #{error}"
    end
  end
end
